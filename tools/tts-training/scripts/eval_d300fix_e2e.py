"""Evaluate D300fixE2E (fixed durations, end-to-end predictor training) on holdout_20.

Evaluates both GT-variance and predicted-variance inference paths.
"""
import argparse, json, math, sys, time
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
OUT = ROOT / "output" / "fixe2e_eval"

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


def run_gt_var(model, phone_ids, dur_gt_t, pitch_gt, energy_gt_t):
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


def run_pred_var(model, phone_ids):
    with torch.no_grad():
        mel, log_dur, pitch, energy = model.forward(phone_ids)
    return mel[0].numpy(), log_dur[0].numpy(), pitch[0].numpy(), energy[0].numpy()


def eval_checkpoint(model, manifest, asr, out_dir, label, limit=0):
    mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
    mel_std = np.array(stats["mel_std"], dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    n = len(manifest) if limit == 0 else min(limit, len(manifest))

    for idx, rec in enumerate(manifest):
        if limit > 0 and idx >= limit:
            break
        pid = rec["poem_id"]
        npz_path = FEAT_DIR / f"{pid}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(str(npz_path))
        mel_gt_raw = npz["mel"].astype(np.float32)
        f0_gt = npz["f0"].astype(np.float32)
        e_gt = npz["energy"].astype(np.float32)
        dur_gt = np.array(rec["durations"], dtype=np.int32)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
        dur_gt_t = torch.tensor(np.array([dur_gt]), dtype=torch.long)
        f0_norm = np.where(
            f0_gt > 0,
            (np.log(np.maximum(f0_gt, 1)) - stats["f0_mean"]) / stats["f0_std"],
            0.0,
        ).astype(np.float32)
        e_norm = ((e_gt - stats["energy_mean"]) / stats["energy_std"]).astype(np.float32)
        pitch_gt = torch.tensor(f0_norm).unsqueeze(0)
        energy_gt_t = torch.tensor(e_norm).unsqueeze(0)
        full_text = rec["text"]

        # --- GT variance path ---
        mel_gtvar_norm = run_gt_var(model, phone_ids, dur_gt_t, pitch_gt, energy_gt_t)
        mel_gtvar_raw = mel_gtvar_norm * mel_std + mel_mean
        T_gt = min(mel_gtvar_norm.shape[0], mel_gt_raw.shape[0])

        audio_gtvar = voc_sess.run(
            None, {"logmel": mel_gtvar_raw[:T_gt].astype(np.float32)}
        )[0].flatten()
        wav_gtvar = str(out_dir / f"{pid}_gtvar.wav")
        sf.write(wav_gtvar, audio_gtvar, 24000)
        asr_gtvar, _ = transcribe(asr, wav_gtvar)
        cer_gtvar, _, _, _, _ = cer_detail(full_text, asr_gtvar)

        # --- Predicted variance path ---
        mel_predvar_norm, pred_log_dur, pred_pitch, pred_energy = run_pred_var(
            model, phone_ids
        )
        mel_predvar_raw = mel_predvar_norm * mel_std + mel_mean
        T_pred = mel_predvar_norm.shape[0]

        audio_predvar = voc_sess.run(
            None, {"logmel": mel_predvar_raw[:T_pred].astype(np.float32)}
        )[0].flatten()
        wav_predvar = str(out_dir / f"{pid}_predvar.wav")
        sf.write(wav_predvar, audio_predvar, 24000)
        asr_predvar, _ = transcribe(asr, wav_predvar)
        cer_predvar, _, _, _, _ = cer_detail(full_text, asr_predvar)

        # --- P2: GT mel reconstruction baseline ---
        audio_p2 = voc_sess.run(
            None, {"logmel": mel_gt_raw[:T_gt].astype(np.float32)}
        )[0].flatten()
        wav_p2 = str(out_dir / f"{pid}_p2gt.wav")
        sf.write(wav_p2, audio_p2, 24000)
        asr_p2, _ = transcribe(asr, wav_p2)
        cer_p2, _, _, _, _ = cer_detail(full_text, asr_p2)

        # Duration stats
        pred_durations = np.maximum(np.round(np.exp(pred_log_dur)), 0).astype(int)
        dur_l1 = float(np.abs(pred_durations - dur_gt).mean())
        pred_dur_total = int(pred_durations.sum())
        gt_dur_total = int(dur_gt.sum())

        speed = (idx + 1) / max(time.time() - t0, 1)
        if idx % 5 == 0 or idx < 3:
            print(
                f"  [{idx+1}/{n}] {pid}: "
                f"P2={cer_p2:.0%} GTvar={cer_gtvar:.0%} PredVar={cer_predvar:.0%} "
                f"(gap={cer_predvar-cer_gtvar:+.0%}) "
                f"dur_l1={dur_l1:.1f} ({gt_dur_total}->{pred_dur_total} frames) "
                f"| {speed:.1f} samp/s"
            )

        results.append(
            {
                "poem_id": pid,
                "cer_p2": cer_p2,
                "cer_gtvar": cer_gtvar,
                "cer_predvar": cer_predvar,
                "predvar_gap": cer_predvar - cer_gtvar,
                "dur_l1": dur_l1,
                "gt_dur_total": gt_dur_total,
                "pred_dur_total": pred_dur_total,
                "full_text": full_text,
            }
        )

    if not results:
        return None

    cers_p2 = [r["cer_p2"] for r in results]
    cers_gt = [r["cer_gtvar"] for r in results]
    cers_pv = [r["cer_predvar"] for r in results]
    gaps = [r["predvar_gap"] for r in results]
    dur_l1s = [r["dur_l1"] for r in results]

    s = {
        "n": len(results),
        "cer_p2_mean": float(np.mean(cers_p2)),
        "cer_gtvar_mean": float(np.mean(cers_gt)),
        "cer_predvar_mean": float(np.mean(cers_pv)),
        "gap_mean": float(np.mean(gaps)),
        "gap_median": float(np.median(gaps)),
        "dur_l1_mean": float(np.mean(dur_l1s)),
        "predvar_100pct": sum(1 for c in cers_pv if c >= 0.99),
        "gtvar_100pct": sum(1 for c in cers_gt if c >= 0.99),
    }
    print(f"\n  [{label}] Summary ({s['n']} samples):")
    print(f"    P2 (GT recon):     CER={s['cer_p2_mean']:.1%}")
    print(f"    GT-var:            CER={s['cer_gtvar_mean']:.1%}  (100%={s['gtvar_100pct']}/{s['n']})")
    print(f"    Pred-var:          CER={s['cer_predvar_mean']:.1%}  (100%={s['predvar_100pct']}/{s['n']})")
    print(f"    Gap (Pred-GT):     mean={s['gap_mean']:+.1%}  median={s['gap_median']:+.1%}")
    print(f"    Duration L1:       {s['dur_l1_mean']:.1f} frames/phone")

    rpt = {"name": label, "summary": s, "samples": results}
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(rpt, f, ensure_ascii=False, indent=2)
    return rpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_labels", nargs="+", default=["6k", "12k", "18k", "24k"])
    parser.add_argument("--limit", type=int, default=0)
    a = parser.parse_args()

    step_map = {
        "6k": "D300fixE2E_step6000_slim.pt",
        "12k": "D300fixE2E_step12000_slim.pt",
        "18k": "D300fixE2E_step18000_slim.pt",
        "24k": "D300fixE2E_step24000_slim.pt",
    }

    holdout = [
        json.loads(l)
        for l in open(ROOT / "data/holdout_20_manifest.jsonl", encoding="utf-8")
    ]
    holdout.sort(key=lambda r: r["mel_len"])

    print("Loading ASR...")
    asr = load_asr_model()
    print("ASR loaded.\n")

    all_reports = {}
    for label in a.step_labels:
        ckpt_path = ROOT / "checkpoints" / step_map[label]
        if not ckpt_path.exists():
            print(f"SKIP {label}: not found")
            continue
        try:
            torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"SKIP {label}: corrupt ({e})")
            continue

        print(f"\n{'='*70}")
        print(f"D300FIX-E2E HOLDOUT_20 @ {label}")
        print(f"{'='*70}")
        model = load_model(str(ckpt_path))
        out_dir = OUT / f"D300fixE2E_holdout20_{label}"
        r = eval_checkpoint(model, holdout, asr, out_dir, f"D300fixE2E_{label}", a.limit)
        all_reports[label] = r

    print(f"\n{'='*70}")
    print("D300FIX-E2E PREDICTED-VARIANCE CURVE")
    print(f"{'='*70}")
    print(
        f"{'Step':>5} {'P2_CER':>7} {'GTvar_CER':>9} {'PredVar_CER':>11} "
        f"{'Gap':>7} {'DurL1':>6} {'PV_100%':>7} {'GV_100%':>7}"
    )
    print("-" * 65)
    for label in a.step_labels:
        if label not in all_reports or all_reports[label] is None:
            continue
        s = all_reports[label]["summary"]
        print(
            f"{label:>5} {s['cer_p2_mean']:>6.1%} {s['cer_gtvar_mean']:>8.1%} "
            f"{s['cer_predvar_mean']:>10.1%} {s['gap_mean']:>+6.1%} "
            f"{s['dur_l1_mean']:>5.1f} {s['predvar_100pct']:>3}/{s['n']:>3} "
            f"{s['gtvar_100pct']:>3}/{s['n']:>3}"
        )

    print("\nDONE")


if __name__ == "__main__":
    main()
