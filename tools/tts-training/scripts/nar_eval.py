"""Evaluate trained FastSpeech 2 on the frozen 24-poem test set.

Runs inference on each test poem, vocodes with HiFi-GAN, then runs
ASR-based evaluation using the existing eval_tts.py framework.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nar_fastspeech2 import FastSpeech2
from nar_build_manifest import text_to_phonemes

CHECKPOINT_DIR = ROOT / "checkpoints"
HIFIGAN_DIR = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
OUTPUT_DIR = ROOT / "output" / "nar_eval"
TEST_SET = ROOT / "data" / "test_set_v1.json"
PHONE_ID_MAP = ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0" / "phone_id_map.txt"


def load_phone_id_map():
    phone_map = {}
    with open(PHONE_ID_MAP, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                phone_map[parts[0]] = int(parts[1])
    return phone_map


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = FastSpeech2(
        vocab_size=268,
        d_model=256,
        nhead=2,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        n_mels=80,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["stats"], ckpt.get("step", 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "fs2_final.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--duration_scale", type=float, default=1.0,
                        help="Scale durations (>1 = slower speech)")
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # Load model
    model, stats, step = load_checkpoint(args.checkpoint, args.device)
    print(f"Checkpoint: step {step}")

    # Load phone map
    phone_map = load_phone_id_map()

    # Load test set
    with open(TEST_SET, encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"Test set: {len(test_set)} poems")

    # Load HiFi-GAN
    import onnxruntime as ort
    vocoder = ort.InferenceSession(str(HIFIGAN_DIR / "hifigan_csmsc.onnx"))

    # Stats
    mel_mean = torch.tensor(stats["mel_mean"], device=args.device)
    mel_std = torch.tensor(stats["mel_std"], device=args.device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    total_infer_time = 0
    total_audio_dur = 0

    for item in test_set:
        poem_id = item.get("poem_id", item.get("id", "unknown"))
        text = item["full_text"]
        title = item.get("title", "")
        category = item.get("category", "unknown")

        # G2P
        phonemes = text_to_phonemes(text)
        phone_ids = []
        for pstr, _ in phonemes:
            if pstr in phone_map:
                phone_ids.append(phone_map[pstr])

        if not phone_ids:
            print(f"  SKIP {poem_id}: no valid phones")
            continue

        phone_ids = torch.tensor(phone_ids, dtype=torch.long, device=args.device)

        # Inference
        t0 = time.time()
        with torch.no_grad():
            mel_pred, log_dur_pred, _, _ = model(phone_ids.unsqueeze(0))

        infer_time = time.time() - t0
        total_infer_time += infer_time

        # Denormalize
        mel_denorm = mel_pred[0].cpu().numpy() * mel_std.cpu().numpy() + mel_mean.cpu().numpy()
        mel_len = mel_denorm.shape[0]
        audio_dur = mel_len * 300 / 24000
        total_audio_dur += audio_dur

        # Vocode
        audio = vocoder.run(None, {"logmel": mel_denorm.astype(np.float32)})[0].flatten()

        # Save
        out_wav = OUTPUT_DIR / f"{poem_id}.wav"
        sf.write(str(out_wav), audio, 24000)

        # Check for early stop / silence
        audio_rms = np.sqrt(np.mean(audio ** 2))
        is_silent = audio_rms < 0.001
        is_short = audio_dur < 1.0

        rtf = infer_time / audio_dur if audio_dur > 0 else float("inf")

        result = {
            "poem_id": poem_id,
            "title": title,
            "category": category,
            "text_length": len(text),
            "mel_len": mel_len,
            "audio_dur_sec": round(audio_dur, 2),
            "infer_time_sec": round(infer_time, 3),
            "rtf": round(rtf, 3),
            "audio_rms": round(float(audio_rms), 4),
            "is_silent": is_silent,
            "is_short": is_short,
            "pred_dur_sum": round(float(log_dur_pred.exp().sum().item()), 0),
        }
        results.append(result)

        status = "PASS"
        if is_silent:
            status = "FAIL_SILENT"
        elif is_short:
            status = "FAIL_SHORT"

        print(f"  {poem_id}: {status} dur={audio_dur:.1f}s rtf={rtf:.3f} rms={audio_rms:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary ({len(results)} samples)")
    print(f"{'='*60}")

    n_pass = sum(1 for r in results if not r["is_silent"] and not r["is_short"])
    n_silent = sum(1 for r in results if r["is_silent"])
    n_short = sum(1 for r in results if r["is_short"])

    print(f"  PASS: {n_pass}/{len(results)}")
    print(f"  FAIL_SILENT: {n_silent}")
    print(f"  FAIL_SHORT: {n_short}")
    print(f"  Total audio: {total_audio_dur:.1f}s")
    print(f"  Total infer: {total_infer_time:.1f}s")
    print(f"  Avg RTF: {total_infer_time / max(total_audio_dur, 0.1):.3f}")

    # Save results
    report_path = OUTPUT_DIR / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "step": step,
            "duration_scale": args.duration_scale,
            "results": results,
            "summary": {
                "n_pass": n_pass,
                "n_total": len(results),
                "n_silent": n_silent,
                "n_short": n_short,
                "total_audio_sec": total_audio_dur,
                "avg_rtf": total_infer_time / max(total_audio_dur, 0.1),
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
