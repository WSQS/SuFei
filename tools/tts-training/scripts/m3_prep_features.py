"""M3 Phase 1: data prep for CosyVoice3 candidates (runs on rtx).

Best-of-N ASR selection, calibrated feature extraction (mel/f0/energy),
manifest assembly, and norm-stats computation.

Follows eval_v10.py conventions (ROOT = E:/sufei-training, ASR via
eval_tts.load_asr_model/transcribe, eval_unified.cer_detail).
Mel extraction reuses the calibrated extract_mel from m3_phase0_assess.py
(magnitude STFT, power=1, log10, eps 1e-10).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("E:/sufei-training")
sys.path.insert(0, str(ROOT / "scripts"))

from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail
from m3_phase0_assess import extract_mel, SR

FRAME_PERIOD = 300.0 / SR * 1000.0  # 12.5 ms — native mel-frame rate


def extract_f0(audio, T):
    """pyworld dio+stonemask at frame_period=12.5 ms (mel frame rate).

    F0 is natively at the mel frame rate — no index-based resampling
    (the historical //256 bug class).  Padded/trimmed to exactly T frames
    (unvoiced = 0).
    """
    import pyworld
    audio_d = audio.astype(np.float64)
    _f0, t = pyworld.dio(
        audio_d, SR,
        frame_period=FRAME_PERIOD,
        f0_floor=80.0, f0_ceil=400.0,
    )
    f0 = pyworld.stonemask(audio_d, _f0, t, SR)
    if len(f0) >= T:
        f0 = f0[:T]
    else:
        f0 = np.pad(f0, (0, T - len(f0)), mode="constant")
    return f0.astype(np.float32)


def compute_energy(mel):
    """Per-frame log10 of linear-domain mel sum.

    np.log10(np.maximum((10.0 ** mel).sum(axis=1), 1e-10))
    Monotone, not floor-dominated.  New dataset => new stats, so the
    definition change vs v3 is safe.
    """
    return np.log10(
        np.maximum((10.0 ** mel).sum(axis=1), 1e-10)
    ).astype(np.float32)


def proportional_durations(n_phonemes, T):
    """Split T as evenly as possible over n_phonemes, summing EXACTLY to T.

    Placeholder for TTSDataset compatibility only (cold-start never expands
    with them; dur_corr diagnostics vs them are meaningless).
    """
    base = T // n_phonemes
    remainder = T % n_phonemes
    return [base + 1] * remainder + [base] * (n_phonemes - remainder)


def extract_features(wav_path):
    """Extract mel, f0, energy from a 24 kHz mono wav file."""
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != SR:
        import librosa
        audio = librosa.resample(
            audio.astype(np.float32), orig_sr=sr, target_sr=SR)
    audio = audio.astype(np.float32)

    mel = extract_mel(audio, SR)   # (T, 80) float32 — calibrated convention
    T = mel.shape[0]
    f0 = extract_f0(audio, T)      # (T,) float32 — native mel-frame rate
    energy = compute_energy(mel)   # (T,) float32
    return mel, f0, energy


def compute_norm_stats(feature_dir, poem_ids):
    """Compute mel/f0/energy norm stats over the given poem set.

    Same JSON schema as train_300_norm_stats_v3.json.
    """
    all_mel = []
    all_f0_voiced = []
    all_energy = []

    for pid in poem_ids:
        npz = np.load(str(feature_dir / f"{pid}.npz"))
        all_mel.append(npz["mel"])
        f0 = npz["f0"]
        voiced = f0[f0 > 0]
        if len(voiced) > 0:
            all_f0_voiced.append(np.log(voiced))
        all_energy.append(npz["energy"])

    mel_cat = np.concatenate(all_mel, axis=0)
    mel_mean = mel_cat.mean(axis=0)
    mel_std = mel_cat.std(axis=0) + 1e-8

    f0_cat = np.concatenate(all_f0_voiced)
    f0_mean = float(f0_cat.mean())
    f0_std = float(f0_cat.std() + 1e-8)

    energy_cat = np.concatenate(all_energy)
    energy_mean = float(energy_cat.mean())
    energy_std = float(energy_cat.std() + 1e-8)

    return {
        "mel_mean": mel_mean.tolist(),
        "mel_std": mel_std.tolist(),
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
    }


def main():
    parser = argparse.ArgumentParser(
        description="M3 Phase 1: feature prep for CosyVoice3 candidates")
    parser.add_argument(
        "--wav_dir",
        default="E:/sufei-training/data/m3_cosyvoice_phase1",
        help="Dir (or comma-separated dirs) with {poem_id}_c{k}.wav "
             "candidates (24 kHz mono)")
    parser.add_argument(
        "--extra_manifest", default=None,
        help="jsonl with {poem_id, text, phoneme_ids}; rows are added to "
             "manifest_recs AND train_ids (all TRAIN)")
    parser.add_argument(
        "--train_manifest",
        default="data/train_300_manifest_v3.jsonl",
        help="Source of poem_id / text / phoneme_ids (train split)")
    parser.add_argument(
        "--holdout_manifest",
        default="data/holdout_20_manifest_v3.jsonl",
        help="Source of poem_id / text / phoneme_ids (holdout split)")
    parser.add_argument("--out_feat_dir", default="data/m3_features")
    parser.add_argument("--out_train", default="data/m3_train_manifest.jsonl")
    parser.add_argument("--out_holdout", default="data/m3_holdout_manifest.jsonl")
    parser.add_argument("--out_stats", default="data/m3_norm_stats.json")
    parser.add_argument("--max_cer", type=float, default=0.45,
                        help="Quality gate: drop poems whose best CER exceeds this")
    args = parser.parse_args()

    wav_dirs = [Path(d) for d in args.wav_dir.split(",")]
    out_feat_dir = ROOT / args.out_feat_dir
    out_feat_dir.mkdir(parents=True, exist_ok=True)

    # ── Load source manifests ──
    manifest_recs = {}  # poem_id -> v3 record
    for mpath in [args.train_manifest, args.holdout_manifest]:
        with open(ROOT / mpath, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                manifest_recs[rec["poem_id"]] = rec

    train_ids = set()
    with open(ROOT / args.train_manifest, encoding="utf-8") as f:
        for line in f:
            train_ids.add(json.loads(line)["poem_id"])

    # ── Extra manifest (phase-2 poems): add to manifest_recs + train_ids ──
    if args.extra_manifest:
        extra_path = ROOT / args.extra_manifest
        extra_count = 0
        with open(extra_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                manifest_recs[rec["poem_id"]] = rec
                train_ids.add(rec["poem_id"])
                extra_count += 1
        print(f"Loaded {extra_count} extra-manifest rows from {extra_path}")

    # ── Group wav candidates by poem_id ──
    candidates = {}  # poem_id -> [(k, Path), ...]
    all_wavs = []
    for wd in wav_dirs:
        all_wavs.extend(wd.glob("*.wav"))
    for wpath in sorted(all_wavs):
        m = re.match(r"^(.+)_c(\d+)$", wpath.stem)
        if not m:
            continue
        pid = m.group(1)
        k = int(m.group(2))
        if pid not in manifest_recs:
            continue
        candidates.setdefault(pid, []).append((k, wpath))

    print(f"Found {len(candidates)} poems with candidate wavs in "
          f"{[str(wd) for wd in wav_dirs]}")

    print("Loading ASR model...")
    asr = load_asr_model()

    # ── Step 1: Best-of-N selection ──
    print("\n=== Best-of-N Selection ===")
    print("%-20s | %5s | %s" % ("poem_id", "best", "CER per candidate"))
    print("-" * 70)

    selected = {}  # poem_id -> (best_k, wav_path, best_cer)
    dropped = []
    for pid in sorted(candidates):
        text = manifest_recs[pid]["text"]
        wav_cands = candidates[pid]

        results = []  # (k, cer, wav_path)
        for k, wpath in sorted(wav_cands):
            asr_text, _ = transcribe(asr, str(wpath))
            cer, _, _, _, _ = cer_detail(text, asr_text)
            results.append((k, cer, wpath))

        # Lowest CER; tie -> lowest k (sort ensures this)
        results.sort(key=lambda r: (r[1], r[0]))
        best_k, best_cer, best_wav = results[0]

        cer_str = ", ".join(f"c{k}={cer*100:.0f}%" for k, cer, _ in results)
        if best_cer > args.max_cer:
            print("%-20s | %5s | %s" % (pid, "DROP", cer_str))
            dropped.append(pid)
        else:
            print("%-20s | c%-4d | %s" % (pid, best_k, cer_str))
            selected[pid] = (best_k, best_wav, best_cer)

    print(f"\nSelected: {len(selected)}  Dropped: {len(dropped)}  "
          f"(max_cer={args.max_cer})")
    if dropped:
        print(f"Dropped poems: {dropped}")

    # ── Step 2: Feature extraction ──
    print("\n=== Feature Extraction ===")
    for pid in sorted(selected):
        _, wav_path, _ = selected[pid]
        mel, f0, energy = extract_features(wav_path)
        np.savez_compressed(
            str(out_feat_dir / f"{pid}.npz"),
            mel=mel, f0=f0, energy=energy,
        )
        print(f"  {pid}: T={mel.shape[0]}, "
              f"f0_voiced={int((f0 > 0).sum())}, "
              f"energy=[{energy.min():.2f}, {energy.max():.2f}]")

    # ── Step 3: Manifest rows ──
    print("\n=== Manifest Assembly ===")
    train_out = ROOT / args.out_train
    holdout_out = ROOT / args.out_holdout
    train_pids = []
    holdout_pids = []

    with open(train_out, "w", encoding="utf-8") as tf, \
         open(holdout_out, "w", encoding="utf-8") as hf:
        for pid in sorted(selected):
            src_rec = manifest_recs[pid]
            _, _, best_cer = selected[pid]
            npz = np.load(str(out_feat_dir / f"{pid}.npz"))
            mel_len = int(npz["mel"].shape[0])
            n_phonemes = len(src_rec["phoneme_ids"])

            row = {
                "poem_id": src_rec["poem_id"],
                "text": src_rec["text"],
                "phoneme_ids": src_rec["phoneme_ids"],
                "durations": proportional_durations(n_phonemes, mel_len),
                "mel_len": mel_len,
                "n_phonemes": n_phonemes,
                "mel_path": f"{args.out_feat_dir}/{pid}.npz",
                "asr_cer": best_cer,
            }
            line = json.dumps(row, ensure_ascii=False)
            if pid in train_ids:
                tf.write(line + "\n")
                train_pids.append(pid)
            else:
                hf.write(line + "\n")
                holdout_pids.append(pid)

    print(f"  Train:   {len(train_pids)} poems -> {train_out}")
    print(f"  Holdout: {len(holdout_pids)} poems -> {holdout_out}")

    # ── Step 4: Norm stats (train split only) ──
    print("\n=== Norm Stats (train split only) ===")
    stats = compute_norm_stats(out_feat_dir, train_pids)
    stats_path = ROOT / args.out_stats
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved -> {stats_path}")
    print(f"  mel_mean[0]={stats['mel_mean'][0]:.4f}  "
          f"f0_mean={stats['f0_mean']:.4f}  "
          f"energy_mean={stats['energy_mean']:.4f}")


if __name__ == "__main__":
    main()
