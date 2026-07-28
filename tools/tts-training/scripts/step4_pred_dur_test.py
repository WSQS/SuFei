"""Step 4: End-to-end test — Pred duration + GT pitch + GT energy → CER.

Uses the log1p duration model (dur_log1p.pt) on top of acoustic baseline.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2, TransformerBlock
from eval_tts import load_asr_model, transcribe, normalize_text, align

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
OUT = ROOT / "output" / "step4"
device = "cpu"


class DurationModel(torch.nn.Module):
    def __init__(self, d_model=256, bias=1.668):
        super().__init__()
        self.adapter = TransformerBlock(d_model, nhead=2, dim_feedforward=512, dropout=0.1)
        self.conv1 = torch.nn.Conv1d(d_model, 256, 3, padding=1)
        self.norm1 = torch.nn.LayerNorm(256)
        self.conv2 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self.norm2 = torch.nn.LayerNorm(256)
        self.linear = torch.nn.Linear(256, 1)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, mask=None):
        x = self.adapter(x, mask=mask)
        x = x.transpose(1, 2)
        x = self.dropout(torch.relu(self.conv1(x)))
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.dropout(torch.relu(self.conv2(x)))
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.dropout(self.linear(x))
        return x.squeeze(-1)


def vocode(mel, sess):
    return sess.run(None, {"logmel": mel.astype(np.float32)})[0].flatten()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Load models
    baseline_ckpt = torch.load(str(ROOT / "checkpoints" / "acoustic_baseline_step12000.pt"),
                               map_location=device, weights_only=False)
    dur_ckpt = torch.load(str(ROOT / "checkpoints" / "dur_log1p.pt"),
                          map_location=device, weights_only=False)

    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(baseline_ckpt["model"])
    model.eval()

    dur_model = DurationModel(d_model=256, bias=dur_ckpt["bias"])
    dur_model.load_state_dict(dur_ckpt["dur_model"])
    dur_model.eval()

    stats = baseline_ckpt["stats"]
    f0_mean, f0_std = stats["f0_mean"], stats["f0_std"]
    e_mean, e_std = stats["energy_mean"], stats["energy_std"]

    voc_sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))

    # Load 12 validation samples
    manifest = [json.loads(l) for l in open(ROOT / "data" / "train_manifest.jsonl", encoding="utf-8")]
    manifest.sort(key=lambda r: r["mel_len"])
    samples = [manifest[i] for i in range(0, 24, 2)]

    print("Loading ASR...")
    asr = load_asr_model()

    results = []
    for rec in samples:
        npz = np.load(str(ROOT / rec["mel_path"]))
        mel_gt = npz["mel"].astype(np.float32)
        f0_gt = npz["f0"].astype(np.float32)
        e_gt = npz["energy"].astype(np.float32)
        dur_gt = np.array(rec["durations"], dtype=np.int32)
        ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long, device=device)

        f0_norm = np.where(f0_gt > 0, (np.log(np.maximum(f0_gt, 1)) - f0_mean) / f0_std, 0.0).astype(np.float32)
        e_norm = ((e_gt - e_mean) / e_std).astype(np.float32)

        content = rec["text"].split("。", 1)[1] if "。" in rec["text"] else rec["text"]
        ref = normalize_text(content)

        with torch.no_grad():
            # Encoder
            x = model.embedding(ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x)

            # Predicted duration
            log_dur = dur_model(x)
            pred_dur = torch.expm1(log_dur).clamp(min=0).round().long()
            dur_sum = pred_dur[0].sum().item()
            ratio = dur_sum / max(dur_gt.sum(), 1)

            # GT pitch/energy expanded using PREDICTED durations
            T_pred = int(pred_dur[0].sum().item())
            mel_input = model.length_regulator(x, pred_dur)

            # Use GT pitch/energy at predicted length
            pitch_gt_full = torch.tensor(f0_norm, device=device).unsqueeze(0)
            energy_gt_full = torch.tensor(e_norm, device=device).unsqueeze(0)

            T_out = mel_input.size(1)
            pitch_use = pitch_gt_full[:, :T_out] if pitch_gt_full.size(1) >= T_out else \
                torch.cat([pitch_gt_full, torch.zeros(1, T_out - pitch_gt_full.size(1))], dim=1)
            energy_use = energy_gt_full[:, :T_out] if energy_gt_full.size(1) >= T_out else \
                torch.cat([energy_gt_full, torch.zeros(1, T_out - energy_gt_full.size(1))], dim=1)

            mel_input = mel_input + \
                model.pitch_embed(pitch_use.unsqueeze(-1)) + \
                model.energy_embed(energy_use.unsqueeze(-1))
            mel_input = model.pos_enc(mel_input)
            dec = mel_input
            for layer in model.decoder_layers:
                dec = layer(dec)
            mel = model.mel_linear(dec)

        mel_np = mel[0].numpy()
        audio = vocode(mel_np, voc_sess)
        wav_path = str(OUT / f"{rec['poem_id']}.wav")
        sf.write(wav_path, audio, 24000)

        asr_text, _ = transcribe(asr, wav_path)
        hyp = normalize_text(asr_text)
        ops = align(ref, hyp)
        d = sum(1 for r, h in ops if r != '*' and h == '*')
        s = sum(1 for r, h in ops if r != '*' and h != '*' and r != h)
        cer = (d + s) / max(len(ref), 1)

        status = "OK" if cer < 0.15 else ("MARGINAL" if cer < 0.3 else "FAIL")
        print(f"  {rec['poem_id']}: ratio={ratio:.0%} CER={cer:.0%} {status} | {content[:25]}")

        results.append({"poem_id": rec["poem_id"], "ratio": ratio, "cer": cer, "status": status})

    # Summary
    print(f"\n{'='*60}")
    print(f"Pred dur + GT pitch + GT energy (12 samples)")
    print(f"{'='*60}")
    med_ratio = np.median([r["ratio"] for r in results])
    med_cer = np.median([r["cer"] for r in results])
    n_ok = sum(1 for r in results if r["status"] == "OK")
    print(f"  Median ratio: {med_ratio:.0%}")
    print(f"  Median CER: {med_cer:.0%}")
    print(f"  Pass: {n_ok}/{len(results)}")


if __name__ == "__main__":
    main()
