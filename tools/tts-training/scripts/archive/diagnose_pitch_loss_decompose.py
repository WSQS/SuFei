"""Decompose frame-level pitch/energy loss to understand the plateau.

Key question: correlation improved 10x but loss barely moved. Why?

Loss = frame-level L1 between pitch_expanded (by pred durations) and f0_norm (GT).
We decompose this into:
  1. Irreducible floor: within-phoneme frame variation (contour/transition)
  2. Duration misalignment: pred dur != GT dur -> frame boundary mismatch
  3. Predictor value error: pred != phoneme_mean

We compute loss under 4 conditions:
  A. pred dur + pred pitch (exact training condition)
  B. GT dur + pred pitch (isolate predictor error)
  C. GT dur + GT phoneme mean (irreducible floor: within-phoneme variation)
  D. GT dur + 0 (always predict global mean baseline)
"""
import json, math, sys, random
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nar_fastspeech2 import FastSpeech2

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "paddle_distill_features"
STATS = json.load(open(ROOT / "data" / "train_300_norm_stats.json"))

f0_mean, f0_std = STATS["f0_mean"], STATS["f0_std"]
e_mean, e_std = STATS["energy_mean"], STATS["energy_std"]

ckpt = torch.load(str(ROOT / "checkpoints" / "FullE2E_v2_step24000_slim.pt"),
                   map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()

train = [json.loads(l) for l in open(ROOT / "data" / "train_300_manifest.jsonl", encoding="utf-8")]

# ============================================================================
# Part 1: Full 300-sample analysis — compute all 4 conditions
# ============================================================================
print("=" * 70)
print("PITCH/ENERGY LOSS DECOMPOSITION (300 samples)")
print("=" * 70)

# Collect frame-level errors under each condition
errors_A_pitch = []  # pred dur + pred pitch (training condition)
errors_B_pitch = []  # GT dur + pred pitch
errors_C_pitch = []  # GT dur + GT phoneme mean (floor)
errors_D_pitch = []  # GT dur + 0 (mean baseline)

errors_A_energy = []
errors_B_energy = []
errors_C_energy = []
errors_D_energy = []

# Also track voiced/unvoiced separately
errors_A_pitch_voiced = []
errors_A_pitch_unvoiced = []
errors_B_pitch_voiced = []
errors_B_pitch_unvoiced = []

dur_errors = []  # |pred_dur - gt_dur| per phoneme

with torch.no_grad():
    for idx, rec in enumerate(train):
        pid = rec["poem_id"]
        npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
        f0 = npz["f0"].astype(np.float32)
        energy = npz["energy"].astype(np.float32)
        durations_gt = np.array(rec["durations"], dtype=int)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
        mel_len = len(f0)

        f0_norm = np.where(f0 > 0,
            (np.log(np.maximum(f0, 1)) - f0_mean) / f0_std, 0.0).astype(np.float32)
        e_norm = ((energy - e_mean) / e_std).astype(np.float32)

        # Forward
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        pred_pitch = model.pitch_predictor(x)[0].numpy()
        pred_energy = model.energy_predictor(x)[0].numpy()
        log_dur_pred = model.duration_predictor(x)[0].numpy()
        pred_durations = np.round(np.exp(log_dur_pred)).clip(min=1).astype(int)

        n_phones = len(durations_gt)

        # Compute phoneme-level GT means (using GT durations)
        gt_pitch_means = np.zeros(n_phones, dtype=np.float32)
        gt_energy_means = np.zeros(n_phones, dtype=np.float32)
        frame_idx = 0
        for i in range(n_phones):
            d = durations_gt[i]
            if d > 0 and frame_idx + d <= mel_len:
                segment_f0 = f0_norm[frame_idx:frame_idx + d]
                segment_e = e_norm[frame_idx:frame_idx + d]
                voiced_mask = f0[frame_idx:frame_idx + d] > 0
                if voiced_mask.any():
                    gt_pitch_means[i] = segment_f0[voiced_mask].mean()
                else:
                    gt_pitch_means[i] = 0.0
                gt_energy_means[i] = segment_e.mean()
            frame_idx += d

        # Condition A: pred dur + pred pitch
        T_A = min(int(pred_durations.sum()), mel_len)
        frame_idx_pred = 0
        for i in range(n_phones):
            d = pred_durations[i]
            for t in range(d):
                f = frame_idx_pred + t
                if f < T_A and f < mel_len:
                    errors_A_pitch.append(abs(pred_pitch[i] - f0_norm[f]))
                    errors_A_energy.append(abs(pred_energy[i] - e_norm[f]))
                    if f0[f] > 0:
                        errors_A_pitch_voiced.append(abs(pred_pitch[i] - f0_norm[f]))
                    else:
                        errors_A_pitch_unvoiced.append(abs(pred_pitch[i] - f0_norm[f]))
            frame_idx_pred += d

        # Condition B: GT dur + pred pitch
        frame_idx = 0
        for i in range(n_phones):
            d = durations_gt[i]
            for t in range(d):
                f = frame_idx + t
                if f < mel_len:
                    errors_B_pitch.append(abs(pred_pitch[i] - f0_norm[f]))
                    errors_B_energy.append(abs(pred_energy[i] - e_norm[f]))
                    if f0[f] > 0:
                        errors_B_pitch_voiced.append(abs(pred_pitch[i] - f0_norm[f]))
                    else:
                        errors_B_pitch_unvoiced.append(abs(pred_pitch[i] - f0_norm[f]))
            frame_idx += d

        # Condition C: GT dur + GT phoneme mean (floor)
        frame_idx = 0
        for i in range(n_phones):
            d = durations_gt[i]
            for t in range(d):
                f = frame_idx + t
                if f < mel_len:
                    errors_C_pitch.append(abs(gt_pitch_means[i] - f0_norm[f]))
                    errors_C_energy.append(abs(gt_energy_means[i] - e_norm[f]))
            frame_idx += d

        # Condition D: GT dur + 0 (always predict global mean)
        frame_idx = 0
        for i in range(n_phones):
            d = durations_gt[i]
            for t in range(d):
                f = frame_idx + t
                if f < mel_len:
                    errors_D_pitch.append(abs(f0_norm[f]))
                    errors_D_energy.append(abs(e_norm[f]))
            frame_idx += d

        # Duration errors
        for i in range(n_phones):
            dur_errors.append(abs(int(pred_durations[i]) - int(durations_gt[i])))

def stats(arr, name):
    arr = np.array(arr)
    print(f"  {name:45s} mean={arr.mean():.4f}  std={arr.std():.4f}")

print("\nPitch L1 decomposition:")
stats(errors_D_pitch, "D: GT dur + predict mean=0 (baseline)")
stats(errors_C_pitch, "C: GT dur + GT phoneme mean (FLOOR)")
stats(errors_B_pitch, "B: GT dur + pred pitch (predictor only)")
stats(errors_A_pitch, "A: pred dur + pred pitch (TRAINING)")

print(f"\n  Floor contribution (C):           {np.mean(errors_C_pitch):.4f}")
print(f"  Predictor error (B - C):         {np.mean(errors_B_pitch) - np.mean(errors_C_pitch):.4f}")
print(f"  Duration misalignment (A - B):   {np.mean(errors_A_pitch) - np.mean(errors_B_pitch):.4f}")

print(f"\n  Voiced/unvoiced breakdown (cond A):")
if errors_A_pitch_voiced:
    print(f"    Voiced frames ({len(errors_A_pitch_voiced)}):   L1={np.mean(errors_A_pitch_voiced):.4f}")
if errors_A_pitch_unvoiced:
    print(f"    Unvoiced frames ({len(errors_A_pitch_unvoiced)}): L1={np.mean(errors_A_pitch_unvoiced):.4f}")
print(f"  Voiced/unvoiced breakdown (cond B):")
if errors_B_pitch_voiced:
    print(f"    Voiced frames ({len(errors_B_pitch_voiced)}):   L1={np.mean(errors_B_pitch_voiced):.4f}")
if errors_B_pitch_unvoiced:
    print(f"    Unvoiced frames ({len(errors_B_pitch_unvoiced)}): L1={np.mean(errors_B_pitch_unvoiced):.4f}")

print("\nEnergy L1 decomposition:")
stats(errors_D_energy, "D: GT dur + predict mean=0 (baseline)")
stats(errors_C_energy, "C: GT dur + GT phoneme mean (FLOOR)")
stats(errors_B_energy, "B: GT dur + pred pitch (predictor only)")
stats(errors_A_energy, "A: pred dur + pred pitch (TRAINING)")

print(f"\n  Floor contribution (C):           {np.mean(errors_C_energy):.4f}")
print(f"  Predictor error (B - C):         {np.mean(errors_B_energy) - np.mean(errors_C_energy):.4f}")
print(f"  Duration misalignment (A - B):   {np.mean(errors_A_energy) - np.mean(errors_B_energy):.4f}")

dur_errors = np.array(dur_errors)
print(f"\nDuration prediction:")
print(f"  Mean |pred - gt|: {dur_errors.mean():.2f} frames")
print(f"  Median:           {np.median(dur_errors):.1f} frames")
print(f"  Exact match rate: {(dur_errors == 0).mean()*100:.1f}%")

# ============================================================================
# Part 2: Within-phoneme F0 contour analysis (why the floor exists)
# ============================================================================
print(f"\n{'=' * 70}")
print("WITHIN-PHONEME F0/ENERGY CONTOUR (floor source)")
print("=" * 70)

within_devs_pitch = []
within_devs_energy = []
for rec in train:
    pid = rec["poem_id"]
    npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
    f0 = npz["f0"].astype(np.float32)
    energy = npz["energy"].astype(np.float32)
    durations = np.array(rec["durations"], dtype=int)
    mel_len = len(f0)

    f0_norm = np.where(f0 > 0,
        (np.log(np.maximum(f0, 1)) - f0_mean) / f0_std, 0.0).astype(np.float32)
    e_norm = ((energy - e_mean) / e_std).astype(np.float32)

    frame_idx = 0
    for d in durations:
        if d > 1 and frame_idx + d <= mel_len:
            seg_f0 = f0_norm[frame_idx:frame_idx + d]
            seg_e = e_norm[frame_idx:frame_idx + d]
            voiced = f0[frame_idx:frame_idx + d] > 0
            if voiced.sum() > 1:
                within_devs_pitch.append(np.abs(seg_f0[voiced] - seg_f0[voiced].mean()).mean())
            if d > 1:
                within_devs_energy.append(np.abs(seg_e - seg_e.mean()).mean())
        frame_idx += d

within_devs_pitch = np.array(within_devs_pitch)
within_devs_energy = np.array(within_devs_energy)
print(f"  Pitch within-phoneme MAD:  mean={within_devs_pitch.mean():.4f}  median={np.median(within_devs_pitch):.4f}")
print(f"  Energy within-phoneme MAD: mean={within_devs_energy.mean():.4f}  median={np.median(within_devs_energy):.4f}")
print(f"  (This is the theoretical floor for frame-level L1 with phoneme-level prediction)")

# ============================================================================
# Part 3: Voiced/unvoiced ratio and its impact
# ============================================================================
print(f"\n{'=' * 70}")
print("VOICED/UNVOICED FRAME ANALYSIS")
print("=" * 70)

total_voiced = 0
total_unvoiced = 0
total_frames = 0
for rec in train:
    pid = rec["poem_id"]
    npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
    f0 = npz["f0"].astype(np.float32)
    v = (f0 > 0).sum()
    total_voiced += v
    total_unvoiced += len(f0) - v
    total_frames += len(f0)

print(f"  Total frames: {total_frames}")
print(f"  Voiced: {total_voiced} ({total_voiced/total_frames*100:.1f}%)")
print(f"  Unvoiced: {total_unvoiced} ({total_unvoiced/total_frames*100:.1f}%)")
print(f"  Unvoiced frames have f0_norm=0 (target).")
print(f"  If predictor outputs nonzero for unvoiced phonemes, each adds |pred| error.")

# Check what predictor outputs for unvoiced phonemes
print(f"\n  Predictor output for unvoiced vs voiced phonemes:")
unvoiced_preds_pitch = []
voiced_preds_pitch = []
unvoiced_preds_energy = []
voiced_preds_energy = []
with torch.no_grad():
    for rec in train[:50]:
        pid = rec["poem_id"]
        npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
        f0 = npz["f0"].astype(np.float32)
        durations = np.array(rec["durations"], dtype=int)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)

        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        pred_p = model.pitch_predictor(x)[0].numpy()
        pred_e = model.energy_predictor(x)[0].numpy()

        frame_idx = 0
        for i, d in enumerate(durations):
            if d > 0 and frame_idx + d <= len(f0):
                seg_voiced = (f0[frame_idx:frame_idx + d] > 0).any()
                if seg_voiced:
                    voiced_preds_pitch.append(pred_p[i])
                    voiced_preds_energy.append(pred_e[i])
                else:
                    unvoiced_preds_pitch.append(pred_p[i])
                    unvoiced_preds_energy.append(pred_e[i])
            frame_idx += d

voiced_preds_pitch = np.array(voiced_preds_pitch)
unvoiced_preds_pitch = np.array(unvoiced_preds_pitch)
voiced_preds_energy = np.array(voiced_preds_energy)
unvoiced_preds_energy = np.array(unvoiced_preds_energy)

print(f"  Pitch pred (voiced):   mean={voiced_preds_pitch.mean():.3f} std={voiced_preds_pitch.std():.3f}")
if len(unvoiced_preds_pitch) > 0:
    print(f"  Pitch pred (unvoiced): mean={unvoiced_preds_pitch.mean():.3f} std={unvoiced_preds_pitch.std():.3f}")
    print(f"  → Unvoiced pitch error contribution: |{unvoiced_preds_pitch.mean():.3f}| × {total_unvoiced/total_frames*100:.1f}% of frames")
print(f"  Energy pred (voiced):   mean={voiced_preds_energy.mean():.3f} std={voiced_preds_energy.std():.3f}")
if len(unvoiced_preds_energy) > 0:
    print(f"  Energy pred (unvoiced): mean={unvoiced_preds_energy.mean():.3f} std={unvoiced_preds_energy.std():.3f}")

# ============================================================================
# Part 4: What loss would perfect prediction give? (with pred dur)
# ============================================================================
print(f"\n{'=' * 70}")
print("SANITY CHECK: what if predictor = GT phoneme mean?")
print("=" * 70)
print("(Using GT durations for expansion, GT phoneme mean as prediction)")
print(f"  This gives the theoretical minimum frame-level L1: {np.mean(errors_C_pitch):.4f} (pitch), {np.mean(errors_C_energy):.4f} (energy)")
print(f"  Training loss plateaus at:                           ~0.53 (pitch), ~0.76 (energy)")
print(f"  Gap = predictor error + duration misalignment")
