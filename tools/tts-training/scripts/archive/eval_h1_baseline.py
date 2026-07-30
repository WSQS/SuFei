"""Stage 1: H1 baseline completion.

1. Full train_171 evaluation (all 171 poems) at 24k
2. holdout_20 curve at 12k / 18k / 24k
3. Save all reports
"""
import json, math, sys, time, numpy as np, torch, soundfile as sf, onnxruntime as ort
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
STATS_PATH = str(ROOT / "data/train_171_norm_stats.json")
FEAT_DIR = str(ROOT / "data/paddle_distill_features")
TEACHER_DIR = str(ROOT / "data/paddle_mfa_corpus")

CKPTS = {
    "12k": str(ROOT / "checkpoints/H1_step12000.pt"),
    "18k": str(ROOT / "checkpoints/H1_step18000.pt"),
    "24k": str(ROOT / "checkpoints/H1_holdout171_24k.pt"),
}

with open(STATS_PATH) as f:
    stats = json.load(f)

voc_sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))


def run_a0_eval(model, manifest, stats, out_dir, asr, label):
    mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
    mel_std = np.array(stats["mel_std"], dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()

    for idx, rec in enumerate(manifest):
        pid = rec["poem_id"]
        npz_path = Path(FEAT_DIR) / f"{pid}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(str(npz_path))
        mel_gt_raw = npz["mel"].astype(np.float32)
        f0_gt = npz["f0"].astype(np.float32)
        e_gt = npz["energy"].astype(np.float32)
        dur_gt = np.array(rec["durations"], dtype=np.int32)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
        dur_gt_t = torch.tensor(np.array([dur_gt]), dtype=torch.long)
        f0_norm = np.where(f0_gt > 0, (np.log(np.maximum(f0_gt, 1)) - stats["f0_mean"]) / stats["f0_std"], 0.0).astype(np.float32)
        e_norm = ((e_gt - stats["energy_mean"]) / stats["energy_std"]).astype(np.float32)
        pitch_gt = torch.tensor(f0_norm).unsqueeze(0)
        energy_gt_t = torch.tensor(e_norm).unsqueeze(0)

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
            mel_model_norm = model.mel_linear(dec)[0].numpy()

        mel_model_raw = mel_model_norm * mel_std + mel_mean
        T = min(mel_model_norm.shape[0], mel_gt_raw.shape[0])
        mel_gt_norm = (mel_gt_raw - mel_mean) / mel_std
        l1 = float(np.abs(mel_model_norm[:T] - mel_gt_norm[:T]).mean())
        full_text = rec["text"]

        # P1
        teacher_wav = Path(TEACHER_DIR) / f"{pid}.wav"
        asr1 = "[SKIP]"
        cer1 = 1.0
        if teacher_wav.exists():
            asr1, _ = transcribe(asr, str(teacher_wav))
            cer1, _, _, _, _ = cer_detail(full_text, asr1)

        # P2
        audio_gt = voc_sess.run(None, {"logmel": mel_gt_raw[:T].astype(np.float32)})[0].flatten()
        wav_gt = str(out_dir / f"{pid}_gt.wav")
        sf.write(wav_gt, audio_gt, 24000)
        asr2, _ = transcribe(asr, wav_gt)
        cer2, _, _, _, _ = cer_detail(full_text, asr2)

        # P3
        audio_model = voc_sess.run(None, {"logmel": mel_model_raw[:T].astype(np.float32)})[0].flatten()
        wav_model = str(out_dir / f"{pid}_model.wav")
        sf.write(wav_model, audio_model, 24000)
        asr3, _ = transcribe(asr, wav_model)
        cer3, _, _, _, _ = cer_detail(full_text, asr3)

        delta = cer3 - cer2
        speed = (idx + 1) / max(time.time() - t0, 1)
        if idx % 10 == 0 or idx < 5:
            print(f"  [{idx+1}/{len(manifest)}] {pid}: L1={l1:.3f} P1={cer1:.0%} P2={cer2:.0%} P3={cer3:.0%} d={delta:+.0%} | {full_text[:25]}")

        results.append({
            "poem_id": pid, "cer1": cer1, "cer2": cer2, "cer3": cer3,
            "student_delta": delta, "mel_l1": l1, "mel_len": rec["mel_len"],
            "full_text": full_text, "asr1": asr1, "asr2": asr2, "asr3": asr3,
        })

    if not results:
        return None

    deltas = [r["student_delta"] for r in results]
    abs_d = [abs(d) for d in deltas]
    l1s = [r["mel_l1"] for r in results]
    cers3 = [r["cer3"] for r in results]

    s = {
        "n": len(results),
        "cer1_mean": float(np.mean([r["cer1"] for r in results])),
        "cer2_mean": float(np.mean([r["cer2"] for r in results])),
        "cer3_mean": float(np.mean(cers3)),
        "delta_mean": float(np.mean(deltas)),
        "delta_median": float(np.median(deltas)),
        "delta_p75": float(np.percentile(deltas, 75)),
        "delta_p90": float(np.percentile(deltas, 90)),
        "abs_delta_le5pp": sum(1 for d in abs_d if d <= 0.05),
        "abs_delta_le5pp_rate": sum(1 for d in abs_d if d <= 0.05) / len(results),
        "p3_100pct": sum(1 for c in cers3 if c >= 0.99),
        "mel_l1_mean": float(np.mean(l1s)),
        "mel_l1_median": float(np.median(l1s)),
    }
    print(f"\n  [{label}] Summary ({s['n']} samples):")
    print(f"    P2={s['cer2_mean']:.1%}  P3={s['cer3_mean']:.1%}")
    print(f"    delta: mean={s['delta_mean']:+.1%}  median={s['delta_median']:+.1%}  P75={s['delta_p75']:+.1%}")
    print(f"    |d|<=5pp: {s['abs_delta_le5pp']}/{s['n']}  P3=100%: {s['p3_100pct']}/{s['n']}")
    print(f"    mel L1: mean={s['mel_l1_mean']:.3f}")

    rpt = {"name": label, "summary": s, "samples": results}
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(rpt, f, ensure_ascii=False, indent=2)
    return rpt


def main():
    # Load manifests
    holdout = [json.loads(l) for l in open(ROOT / "data/holdout_20_manifest.jsonl", encoding="utf-8")]
    holdout.sort(key=lambda r: r["mel_len"])
    train171 = [json.loads(l) for l in open(ROOT / "data/train_171_manifest.jsonl", encoding="utf-8")]
    train171.sort(key=lambda r: r["mel_len"])

    print("Loading ASR...")
    asr = load_asr_model()
    print("ASR loaded.\n")

    # ── Part 1: holdout curve at 12k / 18k / 24k ──
    curve = {}
    for step_label, ckpt_path in CKPTS.items():
        print(f"\n{'='*70}")
        print(f"HOLDOUT_20 @ {step_label}")
        print(f"{'='*70}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = FastSpeech2(vocab_size=268, dropout=0.0)
        model.load_state_dict(ckpt["model"])
        model.eval()
        out_dir = ROOT / "output" / "unified_eval" / f"H1_holdout20_{step_label}"
        r = run_a0_eval(model, holdout, stats, out_dir, asr, f"holdout20_{step_label}")
        curve[step_label] = r

    # ── Part 2: full train_171 @ 24k ──
    print(f"\n{'='*70}")
    print(f"TRAIN_171 FULL @ 24k")
    print(f"{'='*70}")
    ckpt = torch.load(CKPTS["24k"], map_location="cpu", weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()
    out_dir = ROOT / "output" / "unified_eval" / "H1_train171_full"
    r_train = run_a0_eval(model, train171, stats, out_dir, asr, "train171_full_24k")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"HOLDOUT CURVE: 12k → 18k → 24k")
    print(f"{'='*70}")
    print(f"{'Step':>5} {'delta_mean':>12} {'delta_median':>14} {'P75':>8} {'P3=100%':>8} {'L1':>6}")
    print("-" * 55)
    for step_label in ["12k", "18k", "24k"]:
        r = curve[step_label]
        if r:
            s = r["summary"]
            print(f"{step_label:>5} {s['delta_mean']:>+11.1%} {s['delta_median']:>+13.1%} {s['delta_p75']:>+7.1%} {s['p3_100pct']:>4}/{s['n']} {s['mel_l1_mean']:>.3f}")

    print(f"\nTRAIN_171 FULL @ 24k:")
    if r_train:
        s = r_train["summary"]
        print(f"  delta_mean={s['delta_mean']:+.1%}  median={s['delta_median']:+.1%}  |d|<=5pp={s['abs_delta_le5pp']}/{s['n']}  L1={s['mel_l1_mean']:.3f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
