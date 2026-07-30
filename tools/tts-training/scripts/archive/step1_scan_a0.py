"""Step 1: Scan all historical checkpoints for A0 (GT all) CER."""
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
OUT = ROOT / "output" / "ab_matrix"
CKPT_DIR = ROOT / "checkpoints"
POEM_ID = "poem_0285"
POEM_TEXT = "空山不见人，但闻人语响。返景入深林，复照青苔上。"

device = "cpu"


def vocode(mel):
    sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))
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

    # Load sample
    manifest = [json.loads(l) for l in open(ROOT / "data" / "train_manifest.jsonl", encoding="utf-8")]
    rec = next(r for r in manifest if r["poem_id"] == POEM_ID)
    npz = np.load(str(ROOT / rec["mel_path"]))
    dur_gt = np.array(rec["durations"], dtype=np.int32)
    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long, device=device)
    dur_gt_t = torch.tensor(np.array([dur_gt]), dtype=torch.long, device=device)

    with open(ROOT / "data" / "norm_stats.json") as f:
        stats = json.load(f)
    f0_gt = npz["f0"].astype(np.float32)
    energy_gt = npz["energy"].astype(np.float32)
    f0_norm = np.where(f0_gt > 0, (np.log(np.maximum(f0_gt, 1)) - stats["f0_mean"]) / stats["f0_std"], 0.0).astype(np.float32)
    e_norm = ((energy_gt - stats["energy_mean"]) / stats["energy_std"]).astype(np.float32)
    pitch_gt = torch.tensor(f0_norm, device=device).unsqueeze(0)
    energy_gt_t = torch.tensor(e_norm, device=device).unsqueeze(0)

    # Find all checkpoints
    ckpts = sorted(CKPT_DIR.glob("fs2_step*.pt"))
    # Also check fs2_final.pt
    final = CKPT_DIR / "fs2_final.pt"
    if final.exists():
        ckpts.append(final)

    print(f"Found {len(ckpts)} checkpoints to scan\n")

    # Load ASR once
    asr = load_asr_model()
    ref = normalize_text(POEM_TEXT)

    results = []
    for ckpt_path in ckpts:
        name = ckpt_path.stem
        print(f"{'='*50}")
        print(f"Checkpoint: {name}")

        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

        # Try loading with strict=True to detect mismatches
        model = FastSpeech2(vocab_size=268, dropout=0.0)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  WARNING missing keys: {missing[:3]}...")
        if unexpected:
            print(f"  WARNING unexpected keys: {unexpected[:3]}...")
        model.eval()

        step = ckpt.get("step", "?")

        # Duration ratio
        with torch.no_grad():
            x = model.embedding(phone_ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x)
            log_dur = model.duration_predictor(x)
        pred_dur_sum = log_dur[0].exp().sum().item()
        dur_ratio = pred_dur_sum / dur_gt.sum()

        # A0
        mel = run_a0(model, phone_ids, dur_gt_t, pitch_gt, energy_gt_t)
        audio = vocode(mel)
        wav_path = str(OUT / f"a0_{name}.wav")
        sf.write(wav_path, audio, 24000)

        asr_text, _ = transcribe(asr, wav_path)
        hyp = normalize_text(asr_text)
        ops = align(ref, hyp)
        errs = sum(1 for r, h in ops if r != h)
        d = sum(1 for r, h in ops if r != '*' and h == '*')
        s = sum(1 for r, h in ops if r != '*' and h != '*' and r != h)
        i = sum(1 for r, h in ops if r == '*' and h != '*')
        cer = errs / max(len(ref), 1)

        # Content CER (excluding title prefix insertions)
        content_cer = (d + s) / max(len(ref), 1)

        print(f"  step={step}, dur_ratio={dur_ratio:.1%}")
        print(f"  mel range=[{mel.min():.2f}, {mel.max():.2f}]")
        print(f"  A0 CER={cer:.2%} D={d} S={s} I={i}")
        print(f"  Content CER (D+S)={content_cer:.2%}")
        print(f"  ASR: {asr_text}")

        results.append({"name": name, "step": step, "dur_ratio": dur_ratio,
                        "cer": cer, "content_cer": content_cer,
                        "d": d, "s": s, "i": i, "asr": asr_text})

    # Summary table
    print(f"\n{'='*60}")
    print(f"A0 Checkpoint Scan Summary ({POEM_ID})")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<20} {'Step':>6} {'Dur Ratio':>10} {'CER':>8} {'D+S':>6} {'Status'}")
    print("-" * 60)
    for r in results:
        status = "OK" if r["content_cer"] < 0.15 else ("MARGINAL" if r["content_cer"] < 0.3 else "BROKEN")
        print(f"{r['name']:<20} {r['step']:>6} {r['dur_ratio']:>10.1%} {r['cer']:>8.2%} {r['d']+r['s']:>6d}   {status}")


if __name__ == "__main__":
    main()
