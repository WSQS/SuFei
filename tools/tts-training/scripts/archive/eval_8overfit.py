"""8-sample overfit A0 evaluation.

Evaluates the B' 8-sample overfit checkpoint on the SAME 8 training samples.
Uses B' path (GT pitch/energy) which is consistent between training and inference.

This tests whether fixing the LR off-by-one bug allows the model to memorize
8 training samples (target: 8/8 CER < 15%).
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
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe, normalize_text, align

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
OUT = ROOT / "output" / "bprime_8overfit_validation"

device = "cpu"


def vocode(mel, sess):
    return sess.run(None, {"logmel": mel.astype(np.float32)})[0].flatten()


def run_a0(model, phone_ids, dur_gt_t, pitch_gt, energy_gt_t):
    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        mel_input = model.length_regulator(x, dur_gt_t)
        T_out = mel_input.size(1)
        mel_input = mel_input + \
            model.pitch_embed(pitch_gt[:, :T_out].unsqueeze(-1)) + \
            model.energy_embed(energy_gt_t[:, :T_out].unsqueeze(-1))
        mel_input = model.pos_enc(mel_input)
        dec = mel_input
        for layer in model.decoder_layers:
            dec = layer(dec)
        mel = model.mel_linear(dec)
    return mel[0].numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # B' manifest and stats
    manifest = [json.loads(l) for l in open(ROOT / "data" / "paddle_distill_manifest.jsonl", encoding="utf-8")]
    manifest.sort(key=lambda r: r["mel_len"])
    samples = manifest[:8]  # Same 8 samples used in training

    with open(ROOT / "data" / "paddle_distill_norm_stats.json") as f:
        stats = json.load(f)

    voc_sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))

    print("Loading ASR...")
    asr = load_asr_model()

    # Load checkpoint
    ckpt_path = ROOT / "checkpoints" / "fs2_bprime_8overfit.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"\n{'='*60}")
    print(f"B' 8-Sample Overfit Validation (LR bug FIXED)")
    print(f"{'='*60}")

    results = []
    for rec in samples:
        npz_path = ROOT / "data" / "paddle_distill_features" / f"{rec['poem_id']}.npz"
        npz = np.load(str(npz_path))
        mel_gt = npz["mel"].astype(np.float32)
        f0_gt = npz["f0"].astype(np.float32)
        e_gt = npz["energy"].astype(np.float32)
        dur_gt = np.array(rec["durations"], dtype=np.int32)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long, device=device)
        dur_gt_t = torch.tensor(np.array([dur_gt]), dtype=torch.long, device=device)
        f0_norm = np.where(f0_gt > 0, (np.log(np.maximum(f0_gt, 1)) - stats["f0_mean"]) / stats["f0_std"], 0.0).astype(np.float32)
        e_norm = ((e_gt - stats["energy_mean"]) / stats["energy_std"]).astype(np.float32)
        pitch_gt = torch.tensor(f0_norm, device=device).unsqueeze(0)
        energy_gt_t = torch.tensor(e_norm, device=device).unsqueeze(0)

        mel_np = run_a0(model, phone_ids, dur_gt_t, pitch_gt, energy_gt_t)
        audio = vocode(mel_np, voc_sess)
        wav_path = str(OUT / f"bprime_8overfit_{rec['poem_id']}.wav")
        sf.write(wav_path, audio, 24000)

        asr_text, _ = transcribe(asr, wav_path)
        hyp = normalize_text(asr_text)

        full_text = rec["text"]
        content = full_text.split("。", 1)[1] if "。" in full_text else full_text
        ref = normalize_text(content)

        ops = align(ref, hyp)
        subs_d = sum(1 for r, h in ops if r != '*' and h == '*')
        subs_s = sum(1 for r, h in ops if r != '*' and h != '*' and r != h)
        subs_i = sum(1 for r, h in ops if r == '*' and h != '*')
        cer = (subs_d + subs_s) / max(len(ref), 1)

        dur_sec = len(audio) / 24000
        status = "OK" if cer < 0.15 else ("MARGINAL" if cer < 0.3 else "FAIL")

        print(f"  {rec['poem_id']}: CER={cer:.0%} dur={dur_sec:.1f}s {status} | {content[:40]}")
        results.append({"poem_id": rec["poem_id"], "cer": cer, "status": status})

    # Summary
    cers = [x["cer"] for x in results]
    n_pass = sum(1 for x in results if x["status"] == "OK")
    print(f"\n{'='*60}")
    print(f"Summary: Mean CER={np.mean(cers):.1%}, Median CER={np.median(cers):.1%}, Pass: {n_pass}/{len(results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
