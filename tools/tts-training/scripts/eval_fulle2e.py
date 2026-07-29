"""Evaluate FullE2E (predicted durations in training) — train + holdout pred-var CER.

This is the real deployment scenario: no GT durations, no GT pitch/energy.
Tests both train set (30 samples) and holdout (20 samples).
"""
import json, math, sys, time, random
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
STATS_PATH = ROOT / "data" / "train_300_norm_stats.json"
FEAT_DIR = ROOT / "data" / "paddle_distill_features"
OUT = ROOT / "output" / "fulle2e_eval"
OUT.mkdir(parents=True, exist_ok=True)

with open(STATS_PATH) as f:
    stats = json.load(f)

voc_sess = ort.InferenceSession(
    str(HIFIGAN / "hifigan_csmsc.onnx"),
    providers=["CPUExecutionProvider"],
)


def load_model(ckpt_path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def eval_set(model, manifest, asr, out_dir, label):
    mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
    mel_std = np.array(stats["mel_std"], dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()

    for idx, rec in enumerate(manifest):
        pid = rec["poem_id"]
        npz_path = FEAT_DIR / f"{pid}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(str(npz_path))
        mel_gt_raw = npz["mel"].astype(np.float32)
        dur_gt = np.array(rec["durations"], dtype=np.int32)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
        full_text = rec["text"]

        # Pred-var inference (exactly like deployment)
        with torch.no_grad():
            mel_pred, log_dur, pred_pitch, pred_energy = model.forward(phone_ids)

        mel_predvar_norm = mel_pred[0].numpy()
        mel_predvar_raw = mel_predvar_norm * mel_std + mel_mean
        T_pred = mel_predvar_raw.shape[0]

        pred_durations = np.maximum(np.round(np.exp(log_dur[0].numpy())), 0).astype(int)
        dur_l1 = float(np.abs(pred_durations - dur_gt).mean())
        pred_dur_total = int(pred_durations.sum())
        gt_dur_total = int(dur_gt.sum())

        # P2: GT mel reconstruction baseline
        T_gt = mel_gt_raw.shape[0]
        audio_p2 = voc_sess.run(None, {"logmel": mel_gt_raw[:T_gt].astype(np.float32)})[0].flatten()
        wav_p2 = str(out_dir / f"{pid}_p2gt.wav")
        sf.write(wav_p2, audio_p2, 24000)
        asr_p2, _ = transcribe(asr, wav_p2)
        cer_p2, _, _, _, _ = cer_detail(full_text, asr_p2)

        # P3: Model pred-var
        audio_model = voc_sess.run(None, {"logmel": mel_predvar_raw[:T_pred].astype(np.float32)})[0].flatten()
        wav_model = str(out_dir / f"{pid}_model.wav")
        sf.write(wav_model, audio_model, 24000)
        asr_model, _ = transcribe(asr, wav_model)
        cer_model, _, _, _, _ = cer_detail(full_text, asr_model)

        delta = cer_model - cer_p2
        speed = (idx + 1) / max(time.time() - t0, 1)
        if idx % 5 == 0 or idx < 3:
            print(f"  [{idx+1}/{len(manifest)}] {pid}: P2={cer_p2:.0%} Model={cer_model:.0%} "
                  f"d={delta:+.0%} dur={gt_dur_total}->{pred_dur_total} | {full_text[:25]}")

        results.append({
            "poem_id": pid, "cer_p2": cer_p2, "cer_model": cer_model,
            "delta": delta, "dur_l1": dur_l1,
            "pred_dur_total": pred_dur_total, "gt_dur_total": gt_dur_total,
            "full_text": full_text,
        })

    if not results:
        return None

    cers_p2 = [r["cer_p2"] for r in results]
    cers_m = [r["cer_model"] for r in results]
    deltas = [r["delta"] for r in results]
    dur_l1s = [r["dur_l1"] for r in results]

    s = {
        "n": len(results),
        "cer_p2_mean": float(np.mean(cers_p2)),
        "cer_model_mean": float(np.mean(cers_m)),
        "delta_mean": float(np.mean(deltas)),
        "dur_l1_mean": float(np.mean(dur_l1s)),
        "model_100pct": sum(1 for c in cers_m if c >= 0.99),
        "model_lt15": sum(1 for c in cers_m if c < 0.15),
    }
    print(f"\n  [{label}] Summary ({s['n']} samples):")
    print(f"    P2 (GT recon):  CER={s['cer_p2_mean']:.1%}")
    print(f"    Model pred-var: CER={s['cer_model_mean']:.1%}")
    print(f"    Delta:          mean={s['delta_mean']:+.1%}")
    print(f"    100% CER:       {s['model_100pct']}/{s['n']}")
    print(f"    <15% CER:       {s['model_lt15']}/{s['n']}")
    print(f"    Dur L1:         {s['dur_l1_mean']:.1f}")

    return {"name": label, "summary": s, "samples": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_labels", nargs="+", default=["24k"])
    parser.add_argument("--holdout_only", action="store_true")
    a = parser.parse_args()

    step_map = {
        "6k": "FullE2E_step6000_slim.pt",
        "12k": "FullE2E_step12000_slim.pt",
        "18k": "FullE2E_step18000_slim.pt",
        "24k": "FullE2E_step24000_slim.pt",
    }

    holdout = [json.loads(l) for l in open(ROOT / "data/holdout_20_manifest.jsonl", encoding="utf-8")]
    holdout.sort(key=lambda r: r["mel_len"])

    train = [json.loads(l) for l in open(ROOT / "data/train_300_manifest.jsonl", encoding="utf-8")]
    random.seed(42)
    train_sample = random.sample(train, 30)
    train_sample.sort(key=lambda r: r["mel_len"])

    print("Loading ASR...")
    asr = load_asr_model()
    print("ASR loaded.\n")

    for label in a.step_labels:
        ckpt_path = ROOT / "checkpoints" / step_map[label]
        if not ckpt_path.exists():
            print(f"SKIP {label}: not found")
            continue

        print(f"\n{'='*70}")
        print(f"FULL E2E @ {label}")
        print(f"{'='*70}")

        model = load_model(str(ckpt_path))

        if not a.holdout_only:
            r_train = eval_set(model, train_sample, asr,
                               OUT / f"FullE2E_train_{label}", f"FullE2E_train_{label}")

        r_holdout = eval_set(model, holdout, asr,
                             OUT / f"FullE2E_holdout_{label}", f"FullE2E_holdout_{label}")

    print("\nDONE")


if __name__ == "__main__":
    import argparse
    main()
