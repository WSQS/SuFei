"""Verify: is the pitch/energy loss floor caused by within-phoneme variance?

The pitch/energy predictors output 1 scalar per phoneme, but loss is
computed against frame-level GT. If F0/energy varies significantly
within a phoneme (which it must for Mandarin tones), there's an
irreducible L1 error.

This script measures:
1. Within-phoneme std/MAD of normalized F0 and energy
2. Compare to the observed loss floor (~0.55 pitch, ~0.75 energy)
3. Also compute what the loss WOULD be if we averaged GT per phoneme
"""
import json, math, sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "paddle_distill_features"
STATS = json.load(open(ROOT / "data" / "train_300_norm_stats.json"))

f0_mean = STATS["f0_mean"]
f0_std = STATS["f0_std"]
e_mean = STATS["energy_mean"]
e_std = STATS["energy_std"]

train = [json.loads(l) for l in open(ROOT / "data" / "train_300_manifest.jsonl", encoding="utf-8")]

all_within_f0_mad = []   # per-phoneme MAD of normalized F0
all_within_e_mad = []     # per-phoneme MAD of normalized energy
all_within_f0_std = []
all_within_e_std = []
all_phoneme_f0_means = []  # phoneme-level mean F0 (normalized)
all_phoneme_e_means = []
n_phonemes_total = 0
n_voiced_phonemes = 0

for rec in train:
    pid = rec["poem_id"]
    npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
    f0 = npz["f0"].astype(np.float32)
    energy = npz["energy"].astype(np.float32)
    durations = np.array(rec["durations"], dtype=int)

    # Normalize
    f0_norm = np.where(f0 > 0,
        (np.log(np.maximum(f0, 1)) - f0_mean) / f0_std, 0.0).astype(np.float32)
    e_norm = ((energy - e_mean) / e_std).astype(np.float32)

    # Walk through phonemes using durations
    frame_idx = 0
    for dur in durations:
        if dur == 0:
            continue
        start = frame_idx
        end = frame_idx + dur
        frame_idx = end

        phoneme_f0 = f0_norm[start:end]
        phoneme_e = e_norm[start:end]

        # Only consider voiced frames for F0
        voiced = phoneme_f0[f0[start:end] > 0]
        if len(voiced) > 1:
            mad = np.abs(voiced - voiced.mean()).mean()
            std = voiced.std()
            all_within_f0_mad.append(mad)
            all_within_f0_std.append(std)
            all_phoneme_f0_means.append(voiced.mean())
            n_voiced_phonemes += 1

        if len(phoneme_e) > 1:
            mad = np.abs(phoneme_e - phoneme_e.mean()).mean()
            std = phoneme_e.std()
            all_within_e_mad.append(mad)
            all_within_e_std.append(std)
            all_phoneme_e_means.append(phoneme_e.mean())

        n_phonemes_total += 1

f0_mad = np.array(all_within_f0_mad)
e_mad = np.array(all_within_e_mad)
f0_std_arr = np.array(all_within_f0_std)
e_std_arr = np.array(all_within_e_std)

print(f"{'='*70}")
print(f"WITHIN-PHONEME VARIANCE ANALYSIS ({n_phonemes_total} phonemes, {n_voiced_phonemes} voiced)")
print(f"{'='*70}")

print(f"\nF0 (normalized log-Hz):")
print(f"  Within-phoneme MAD:  mean={f0_mad.mean():.4f} median={np.median(f0_mad):.4f}")
print(f"  Within-phoneme std:  mean={f0_std_arr.mean():.4f} median={np.median(f0_std_arr):.4f}")
print(f"  This is the IRREDUCIBLE L1 if predictor outputs phoneme mean")
print(f"  Observed pitch loss floor: ~0.55")
print(f"  Predicted floor (MAD):     {f0_mad.mean():.4f}")
print(f"  -> {'MATCHES' if abs(f0_mad.mean() - 0.55) < 0.15 else 'DOES NOT MATCH'} observed floor")

print(f"\nEnergy (normalized):")
print(f"  Within-phoneme MAD:  mean={e_mad.mean():.4f} median={np.median(e_mad):.4f}")
print(f"  Within-phoneme std:  mean={e_std_arr.mean():.4f} median={np.median(e_std_arr):.4f}")
print(f"  Observed energy loss floor: ~0.75")
print(f"  Predicted floor (MAD):      {e_mad.mean():.4f}")
print(f"  -> {'MATCHES' if abs(e_mad.mean() - 0.75) < 0.15 else 'DOES NOT MATCH'} observed floor")

# Distribution of within-phoneme F0 variation by phoneme length
print(f"\nF0 MAD by phoneme duration (frames):")
durations_bins = [(1, 3), (4, 6), (7, 10), (11, 20), (21, 50), (51, 200)]
for lo, hi in durations_bins:
    # Re-collect with duration info
    mad_in_bin = []
    for rec in train:
        pid = rec["poem_id"]
        npz_path = FEAT_DIR / f"{pid}.npz"
        npz = np.load(str(npz_path))
        f0 = npz["f0"].astype(np.float32)
        durations = np.array(rec["durations"], dtype=int)
        f0_norm = np.where(f0 > 0,
            (np.log(np.maximum(f0, 1)) - f0_mean) / f0_std, 0.0).astype(np.float32)

        frame_idx = 0
        for dur in durations:
            if dur == 0:
                continue
            start = frame_idx
            end = frame_idx + dur
            frame_idx = end
            if lo <= dur <= hi:
                voiced = f0_norm[start:end][f0[start:end] > 0]
                if len(voiced) > 1:
                    mad_in_bin.append(np.abs(voiced - voiced.mean()).mean())

    if mad_in_bin:
        mad_arr = np.array(mad_in_bin)
        print(f"  dur {lo:>3}-{hi:<3} frames: n={len(mad_arr):>5} MAD={mad_arr.mean():.4f} (median={np.median(mad_arr):.4f})")

# What if we used per-phoneme AVERAGE as GT (proper FS2 approach)?
print(f"\n{'='*70}")
print(f"ALTERNATIVE: phoneme-averaged GT loss")
print(f"{'='*70}")
print(f"\nIf we averaged F0/energy per phoneme before computing loss:")
print(f"  The predictor would learn phoneme-level means")
print(f"  F0 mean range: [{np.min(all_phoneme_f0_means):.2f}, {np.max(all_phoneme_f0_means):.2f}]")
print(f"  E  mean range: [{np.min(all_phoneme_e_means):.2f}, {np.max(all_phoneme_e_means):.2f}]")

# The between-phoneme variance is what the predictor CAN learn
f0_between_std = np.std(all_phoneme_f0_means)
e_between_std = np.std(all_phoneme_e_means)
print(f"\n  Between-phoneme std (learnable signal):")
print(f"    F0: {f0_between_std:.4f}")
print(f"    E:  {e_between_std:.4f}")
print(f"  Within-phoneme std (irreducible noise):")
print(f"    F0: {f0_std_arr.mean():.4f}")
print(f"    E:  {e_std_arr.mean():.4f}")
print(f"  Signal-to-noise ratio:")
print(f"    F0: {f0_between_std / f0_std_arr.mean():.2f}x")
print(f"    E:  {e_between_std / e_std_arr.mean():.2f}x")

# How many frames per phoneme on average?
all_durs = []
for rec in train:
    durs = [d for d in rec["durations"] if d > 0]
    all_durs.extend(durs)
print(f"\nAvg frames per phoneme: {np.mean(all_durs):.1f} (median={np.median(all_durs):.0f})")
print(f"This means each phoneme-level prediction is compared against")
print(f"~{np.mean(all_durs):.0f} different frame-level GT values.")

print("\nDONE")
