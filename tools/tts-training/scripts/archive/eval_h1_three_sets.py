"""Evaluate H1 model on train_171, holdout_20, and external_unseen_6.

Runs full P1/P2/P3 with mel L1 for each set.
Uses train_171 norm stats for all evaluations (consistent with H1 training).
"""
import argparse, json, math, sys, time, numpy as np, torch, soundfile as sf, onnxruntime as ort
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
CKPT = str(ROOT / "checkpoints/H1_holdout171_24k.pt")
STATS = str(ROOT / "data/train_171_norm_stats.json")


def evaluate_set(name, manifest_path, feature_dir, teacher_wav_dir, stats, model, voc_sess, asr, limit=0):
    """Evaluate P1/P2/P3 on a set of poems."""
    out_dir = ROOT / "output" / "unified_eval" / f"H1_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
    mel_std = np.array(stats["mel_std"], dtype=np.float32)

    manifest = [json.loads(l) for l in open(manifest_path, encoding="utf-8")]
    manifest.sort(key=lambda r: r["mel_len"])
    if limit > 0:
        manifest = manifest[:limit]

    results = []
    t0 = time.time()

    for idx, rec in enumerate(manifest):
        pid = rec["poem_id"]
        npz_path = Path(feature_dir) / f"{pid}.npz"
        if not npz_path.exists():
            print(f"  SKIP {pid}: no features")
            continue

        npz = np.load(str(npz_path))
        mel_gt_raw = npz["mel"].astype(np.float32)
        f0_gt = npz["f0"].astype(np.float32)
        e_gt = npz["energy"].astype(np.float32)
        dur_gt = np.array(rec["durations"], dtype=np.int32)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
        dur_gt_t = torch.tensor(np.array([dur_gt]), dtype=torch.long)
        f0_norm = np.where(f0_gt > 0,
                           (np.log(np.maximum(f0_gt, 1)) - stats["f0_mean"]) / stats["f0_std"],
                           0.0).astype(np.float32)
        e_norm = ((e_gt - stats["energy_mean"]) / stats["energy_std"]).astype(np.float32)
        pitch_gt = torch.tensor(f0_norm).unsqueeze(0)
        energy_gt_t = torch.tensor(e_norm).unsqueeze(0)

        # A0 inference
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

        # P1: teacher wav
        teacher_wav = Path(teacher_wav_dir) / f"{pid}.wav"
        if teacher_wav.exists():
            asr1, _ = transcribe(asr, str(teacher_wav))
        else:
            asr1 = "[SKIP]"
        cer1, _, _, _, _ = cer_detail(full_text, asr1)

        # P2: GT raw mel
        audio_gt = voc_sess.run(None, {"logmel": mel_gt_raw[:T].astype(np.float32)})[0].flatten()
        wav_gt = str(out_dir / f"{pid}_gt.wav")
        sf.write(wav_gt, audio_gt, 24000)
        asr2, _ = transcribe(asr, wav_gt)
        cer2, _, _, _, _ = cer_detail(full_text, asr2)

        # P3: Model raw mel (denorm)
        audio_model = voc_sess.run(None, {"logmel": mel_model_raw[:T].astype(np.float32)})[0].flatten()
        wav_model = str(out_dir / f"{pid}_model.wav")
        sf.write(wav_model, audio_model, 24000)
        asr3, _ = transcribe(asr, wav_model)
        cer3, _, _, _, _ = cer_detail(full_text, asr3)

        delta = cer3 - cer2
        speed = (idx + 1) / max(time.time() - t0, 1)
        status = "OK" if cer3 < 0.15 else ("MARG" if cer3 < 0.3 else "FAIL")
        print(f"[{idx+1}/{len(manifest)}] {pid}: L1={l1:.3f} P1={cer1:.0%} P2={cer2:.0%} P3={cer3:.0%} {status} "
              f"delta={delta:+.0%} | {full_text[:30]}")

        results.append({
            "poem_id": pid, "cer1": cer1, "cer2": cer2, "cer3": cer3,
            "student_delta": delta, "mel_l1": l1,
            "full_text": full_text,
            "asr1": asr1, "asr2": asr2, "asr3": asr3,
            "mel_len": rec["mel_len"],
        })

    # Summary
    if not results:
        return None

    cers1 = [r["cer1"] for r in results if r["cer1"] >= 0]
    cers2 = [r["cer2"] for r in results]
    cers3 = [r["cer3"] for r in results]
    deltas = [r["student_delta"] for r in results]
    abs_deltas = [abs(d) for d in deltas]
    l1s = [r["mel_l1"] for r in results]

    print(f"\n  Summary: {name} ({len(results)} samples)")
    print(f"    P1 mean={np.mean(cers1):.1%}  P2 mean={np.mean(cers2):.1%}  P3 mean={np.mean(cers3):.1%}")
    print(f"    student_delta: mean={np.mean(deltas):+.4f}  median={np.median(deltas):+.4f}")
    print(f"    P75={np.percentile(deltas, 75):+.4f}  P90={np.percentile(deltas, 90):+.4f}")
    le5 = sum(1 for d in abs_deltas if d <= 0.05)
    print(f"    |delta|<=5pp: {le5}/{len(results)} = {le5/len(results):.0%}")
    print(f"    P3 CER=100%: {sum(1 for c in cers3 if c >= 0.99)}/{len(results)}")
    print(f"    mel L1: mean={np.mean(l1s):.4f}  median={np.median(l1s):.4f}")

    report = {
        "name": f"H1_{name}",
        "n_samples": len(results),
        "summary": {
            "cer1_mean": float(np.mean(cers1)),
            "cer2_mean": float(np.mean(cers2)),
            "cer3_mean": float(np.mean(cers3)),
            "student_delta_mean": float(np.mean(deltas)),
            "student_delta_median": float(np.median(deltas)),
            "student_delta_p75": float(np.percentile(deltas, 75)),
            "student_delta_p90": float(np.percentile(deltas, 90)),
            "abs_delta_le5pp": le5,
            "abs_delta_le5pp_rate": le5 / len(results),
            "p3_100pct_count": sum(1 for c in cers3 if c >= 0.99),
            "mel_l1_mean": float(np.mean(l1s)),
            "mel_l1_median": float(np.median(l1s)),
        },
        "samples": results,
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"    Report: {out_dir / 'report.json'}")
    return report


def main():
    print("Loading model...")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with open(STATS) as f:
        stats = json.load(f)

    voc_sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))

    print("Loading ASR...")
    asr = load_asr_model()
    print("ASR loaded.\n")

    feat_dir = str(ROOT / "data/paddle_distill_features")
    teacher_dir = str(ROOT / "data/paddle_mfa_corpus")
    unseen_feat_dir = str(ROOT / "data/unseen_features")
    unseen_teacher_dir = str(ROOT / "output/unseen_teacher_wav")

    # 1. Train 171 (limit to 50 for speed)
    print(f"{'='*70}")
    print("SET 1: train_171 (sample 50)")
    print(f"{'='*70}")
    r1 = evaluate_set("train171_sample", str(ROOT / "data/train_171_manifest.jsonl"),
                       feat_dir, teacher_dir, stats, model, voc_sess, asr, limit=50)

    # 2. Holdout 20 (full)
    print(f"\n{'='*70}")
    print("SET 2: holdout_20 (full)")
    print(f"{'='*70}")
    r2 = evaluate_set("holdout20", str(ROOT / "data/holdout_20_manifest.jsonl"),
                       feat_dir, teacher_dir, stats, model, voc_sess, asr)

    # 3. External unseen 6 (full)
    print(f"\n{'='*70}")
    print("SET 3: external_unseen_6 (full)")
    print(f"{'='*70}")
    r3 = evaluate_set("external6", str(ROOT / "data/external_unseen_6_manifest.jsonl"),
                       unseen_feat_dir, unseen_teacher_dir, stats, model, voc_sess, asr)

    # Cross-set comparison
    print(f"\n{'='*70}")
    print("CROSS-SET COMPARISON")
    print(f"{'='*70}")
    print(f"{'Set':<25} {'n':>3} {'P2':>6} {'P3':>6} {'delta':>8} {'P75':>8} {'L1':>6} {'100%':>5}")
    print("-" * 70)
    for name, r in [("train_171 (sample 50)", r1), ("holdout_20", r2), ("external_unseen_6", r3)]:
        if r:
            s = r["summary"]
            print(f"{name:<25} {r['n_samples']:>3} {s['cer2_mean']:>5.1%} {s['cer3_mean']:>5.1%} "
                  f"{s['student_delta_mean']:>+7.1%} {s['student_delta_p75']:>+7.1%} "
                  f"{s['mel_l1_mean']:>.3f} {s['p3_100pct_count']:>5}")


if __name__ == "__main__":
    main()
