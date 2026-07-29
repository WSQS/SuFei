"""Unified evaluation with proper denormalization.

For each sample, evaluates 4 paths:
  P1: Teacher wav → ASR           (teacher/ASR baseline)
  P2: GT raw mel → HiFiGAN → ASR  (mel-vocoder reconstruction)
  P3: Model raw mel → HiFiGAN → ASR (student A0, denormalized)
  P4: Model norm mel → HiFiGAN → ASR (old buggy path, for comparison)

Reports per-sample CER + deltas:
  reconstruction_delta = P2 - P1
  student_delta        = P3 - P2
  denorm_bug           = P4 - P3

Usage:
  python eval_unified.py --ckpt checkpoints/fs2_bprime_final.pt \
    --manifest data/paddle_distill_manifest.jsonl \
    --features paddle_distill_features \
    --stats data/paddle_distill_norm_stats.json \
    --teacher_wav_dir data/paddle_mfa_corpus \
    --name B_prime_191
"""
import argparse
import json
import math
import sys
import time
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
OUT = ROOT / "output" / "unified_eval"

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


def cer_detail(ref_text, hyp_text):
    ref = normalize_text(ref_text)
    hyp = normalize_text(hyp_text)
    if len(ref) == 0:
        return 0.0, 0, 0, 0, 0
    ops = align(ref, hyp)
    subs_d = sum(1 for r, h in ops if r != '*' and h == '*')
    subs_s = sum(1 for r, h in ops if r != '*' and h != '*' and r != h)
    subs_i = sum(1 for r, h in ops if r == '*' and h != '*')
    cer = (subs_d + subs_s) / max(len(ref), 1)
    return cer, subs_d, subs_s, subs_i, len(ref)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--teacher_wav_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0=all)")
    parser.add_argument("--skip_teacher", action="store_true", help="Skip teacher wav path")
    args = parser.parse_args()

    out_dir = OUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_path = ROOT / args.stats if not Path(args.stats).is_absolute() else Path(args.stats)
    with open(stats_path) as f:
        stats = json.load(f)
    mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
    mel_std = np.array(stats["mel_std"], dtype=np.float32)

    voc_sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))

    print("Loading ASR...")
    asr = load_asr_model()

    ckpt_path = Path(args.ckpt) if Path(args.ckpt).is_absolute() else ROOT / args.ckpt
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    manifest_path = ROOT / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)
    manifest = [json.loads(l) for l in open(manifest_path, encoding="utf-8")]
    manifest.sort(key=lambda r: r["mel_len"])
    if args.limit > 0:
        manifest = manifest[:args.limit]

    teacher_dir = ROOT / args.teacher_wav_dir if not Path(args.teacher_wav_dir).is_absolute() else Path(args.teacher_wav_dir)

    results = []
    t_start = time.time()

    for idx, rec in enumerate(manifest):
        target_id = rec["poem_id"]
        features_path = ROOT / args.features if not Path(args.features).is_absolute() else Path(args.features)
        npz_path = features_path / f"{target_id}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(str(npz_path))

        mel_gt_raw = npz["mel"].astype(np.float32)
        f0_gt = npz["f0"].astype(np.float32)
        e_gt = npz["energy"].astype(np.float32)
        dur_gt = np.array(rec["durations"], dtype=np.int32)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long, device=device)
        dur_gt_t = torch.tensor(np.array([dur_gt]), dtype=torch.long, device=device)
        f0_norm = np.where(f0_gt > 0, (np.log(np.maximum(f0_gt, 1)) - stats["f0_mean"]) / stats["f0_std"], 0.0).astype(np.float32)
        e_norm = ((e_gt - stats["energy_mean"]) / stats["energy_std"]).astype(np.float32)
        pitch_gt = torch.tensor(f0_norm, device=device).unsqueeze(0)
        energy_gt_t = torch.tensor(e_norm, device=device).unsqueeze(0)

        mel_model_norm = run_a0(model, phone_ids, dur_gt_t, pitch_gt, energy_gt_t)
        mel_model_raw = mel_model_norm * mel_std + mel_mean

        full_text = rec["text"]
        T = min(mel_model_norm.shape[0], mel_gt_raw.shape[0])

        # ── Path 1: Teacher wav → ASR ──
        teacher_wav = teacher_dir / f"{target_id}.wav"
        if teacher_wav.exists() and not args.skip_teacher:
            asr_text_1, _ = transcribe(asr, str(teacher_wav))
        else:
            asr_text_1 = "[SKIP]"

        # ── Path 2: GT raw mel → HiFiGAN → ASR ──
        audio_gt = vocode(mel_gt_raw[:T], voc_sess)
        wav_gt = str(out_dir / f"{target_id}_gt.wav")
        sf.write(wav_gt, audio_gt, 24000)
        asr_text_2, _ = transcribe(asr, wav_gt)

        # ── Path 3: Model raw mel (denorm) → HiFiGAN → ASR ──
        audio_model = vocode(mel_model_raw[:T], voc_sess)
        wav_model = str(out_dir / f"{target_id}_model.wav")
        sf.write(wav_model, audio_model, 24000)
        asr_text_3, _ = transcribe(asr, wav_model)

        cer1, _, _, _, _ = cer_detail(full_text, asr_text_1)
        cer2, _, _, _, _ = cer_detail(full_text, asr_text_2)
        cer3, _, _, _, _ = cer_detail(full_text, asr_text_3)

        recon_delta = cer2 - cer1
        student_delta = cer3 - cer2

        elapsed = time.time() - t_start
        speed = (idx + 1) / elapsed

        status = "OK" if cer3 < 0.15 else ("MARG" if cer3 < 0.3 else "FAIL")
        print(f"[{idx+1}/{len(manifest)}] {target_id}: "
              f"P1={cer1:.0%} P2={cer2:.0%} P3={cer3:.0%} {status} | "
              f"recon={recon_delta:+.0%} student={student_delta:+.0%} | "
              f"{speed:.1f} samp/s | {full_text[:25]}")

        results.append({
            "poem_id": target_id,
            "cer1": cer1, "cer2": cer2, "cer3": cer3,
            "recon_delta": recon_delta, "student_delta": student_delta,
            "full_text": full_text,
            "asr1": asr_text_1, "asr2": asr_text_2, "asr3": asr_text_3,
            "mel_len": rec["mel_len"],
        })

    # ── Summary ──
    cers1 = [r["cer1"] for r in results if r["cer1"] >= 0]
    cers2 = [r["cer2"] for r in results]
    cers3 = [r["cer3"] for r in results]
    recon_deltas = [r["recon_delta"] for r in results if r["cer1"] >= 0]
    student_deltas = [r["student_delta"] for r in results]

    n_pass = sum(1 for c in cers3 if c < 0.15)
    n_marg = sum(1 for c in cers3 if 0.15 <= c < 0.3)

    print(f"\n{'='*70}")
    print(f"Summary: {args.name} ({len(results)} samples)")
    print(f"{'='*70}")
    print(f"{'Path':<30} {'Mean CER':>10} {'Median':>10} {'Pass<15%':>9}")
    print("-" * 62)
    if cers1:
        print(f"{'P1 Teacher wav':<30} {np.mean(cers1):>10.1%} {np.median(cers1):>10.1%} {sum(1 for c in cers1 if c<0.15):>4}/{len(cers1)}")
    print(f"{'P2 GT mel reconstruction':<30} {np.mean(cers2):>10.1%} {np.median(cers2):>10.1%} {sum(1 for c in cers2 if c<0.15):>4}/{len(cers2)}")
    print(f"{'P3 Model mel (denorm)':<30} {np.mean(cers3):>10.1%} {np.median(cers3):>10.1%} {n_pass:>4}/{len(cers3)}")
    print(f"\n{'Mean deltas':<30}")
    if recon_deltas:
        print(f"{'  reconstruction (P2-P1)':<30} {np.mean(recon_deltas):>+10.1%}")
    print(f"{'  student (P3-P2)':<30} {np.mean(student_deltas):>+10.1%}")

    # Save JSON
    report = {
        "name": args.name,
        "n_samples": len(results),
        "summary": {
            "cer1_mean": float(np.mean(cers1)) if cers1 else None,
            "cer2_mean": float(np.mean(cers2)),
            "cer3_mean": float(np.mean(cers3)),
            "recon_delta_mean": float(np.mean(recon_deltas)) if recon_deltas else None,
            "student_delta_mean": float(np.mean(student_deltas)),
            "n_pass": n_pass,
            "n_marginal": n_marg,
        },
        "samples": results,
    }
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
