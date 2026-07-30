"""Why can't pitch/energy predictors learn when the signal is clear?

Within-phoneme F0 MAD = 0.058, but loss floor = 0.55.
The predictor should easily learn phoneme-level means (SNR = 15.9x).
Yet it doesn't. Why?

Hypotheses to test:
1. Duration expansion: most phonemes have dur=1-2 frames (median=2),
   but some have dur=20+. Long phonemes dominate frame-level L1 loss.
2. Loss weighting: long phonemes contribute more frames => more loss,
   but the predictor sees the same 1 scalar regardless of duration.
3. Actually check: what does the model PREDICT vs GT for pitch/energy?
   Is it outputting a constant? Is it outputting noise? Is it stuck?
4. Check if the predictor gradient is actually flowing (maybe detached?)
5. Check the loss computation path for full_e2e mode specifically.
"""
import json, math, sys, numpy as np, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nar_fastspeech2 import FastSpeech2

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "paddle_distill_features"
STATS = json.load(open(ROOT / "data" / "train_300_norm_stats.json"))

f0_mean = STATS["f0_mean"]
f0_std = STATS["f0_std"]
e_mean = STATS["energy_mean"]
e_std = STATS["energy_std"]

ckpt = torch.load(str(ROOT / "checkpoints" / "FullE2E_v2_step24000_slim.pt"),
                   map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()

train = [json.loads(l) for l in open(ROOT / "data" / "train_300_manifest.jsonl", encoding="utf-8")]

# ============================================================================
# 1. Direct comparison: predictor output vs phoneme-averaged GT
# ============================================================================
import random
random.seed(42)
sample = random.sample(train, 30)

all_pred_pitch = []
all_gt_pitch_mean = []  # phoneme-averaged GT
all_pred_energy = []
all_gt_energy_mean = []
all_durations = []

for rec in sample:
    pid = rec["poem_id"]
    npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
    f0 = npz["f0"].astype(np.float32)
    energy = npz["energy"].astype(np.float32)
    durations = np.array(rec["durations"], dtype=int)
    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)

    f0_norm = np.where(f0 > 0,
        (np.log(np.maximum(f0, 1)) - f0_mean) / f0_std, 0.0).astype(np.float32)
    e_norm = ((energy - e_mean) / e_std).astype(np.float32)

    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        pred_pitch = model.pitch_predictor(x)[0].numpy()
        pred_energy = model.energy_predictor(x)[0].numpy()

    # Compute phoneme-level GT means
    frame_idx = 0
    for i, dur in enumerate(durations):
        if dur == 0:
            continue
        start = frame_idx
        end = frame_idx + dur
        frame_idx = end

        voiced = f0_norm[start:end][f0[start:end] > 0]
        if len(voiced) > 0:
            all_pred_pitch.append(pred_pitch[i])
            all_gt_pitch_mean.append(voiced.mean())
        else:
            all_pred_pitch.append(pred_pitch[i])
            all_gt_pitch_mean.append(0.0)  # unvoiced

        all_pred_energy.append(pred_energy[i])
        all_gt_energy_mean.append(e_norm[start:end].mean())
        all_durations.append(dur)

all_pred_pitch = np.array(all_pred_pitch)
all_gt_pitch_mean = np.array(all_gt_pitch_mean)
all_pred_energy = np.array(all_pred_energy)
all_gt_energy_mean = np.array(all_gt_energy_mean)
all_durations = np.array(all_durations)

print(f"{'='*70}")
print(f"PREDICTOR OUTPUT vs GT (phoneme-level, {len(all_pred_pitch)} phonemes)")
print(f"{'='*70}")

print(f"\nPitch (normalized log-F0):")
print(f"  Pred:  mean={all_pred_pitch.mean():.3f} std={all_pred_pitch.std():.3f} range=[{all_pred_pitch.min():.2f}, {all_pred_pitch.max():.2f}]")
print(f"  GT:    mean={all_gt_pitch_mean.mean():.3f} std={all_gt_pitch_mean.std():.3f} range=[{all_gt_pitch_mean.min():.2f}, {all_gt_pitch_mean.max():.2f}]")
l1_phoneme = np.abs(all_pred_pitch - all_gt_pitch_mean).mean()
print(f"  L1 (phoneme-level): {l1_phoneme:.4f}")
print(f"  Correlation: {np.corrcoef(all_pred_pitch, all_gt_pitch_mean)[0,1]:.4f}")

print(f"\nEnergy (normalized):")
print(f"  Pred:  mean={all_pred_energy.mean():.3f} std={all_pred_energy.std():.3f} range=[{all_pred_energy.min():.2f}, {all_pred_energy.max():.2f}]")
print(f"  GT:    mean={all_gt_energy_mean.mean():.3f} std={all_gt_energy_mean.std():.3f} range=[{all_gt_energy_mean.min():.2f}, {all_gt_energy_mean.max():.2f}]")
l1_e_phoneme = np.abs(all_pred_energy - all_gt_energy_mean).mean()
print(f"  L1 (phoneme-level): {l1_e_phoneme:.4f}")
print(f"  Correlation: {np.corrcoef(all_pred_energy, all_gt_energy_mean)[0,1]:.4f}")

# ============================================================================
# 2. Is the predictor outputing near-constant values?
# ============================================================================
print(f"\n{'='*70}")
print(f"IS THE PREDICTOR OUTPUTTING CONSTANTS?")
print(f"{'='*70}")

# What would "predict the mean" L1 be?
print(f"\n'Always predict global mean' baseline:")
print(f"  Pitch L1: {np.abs(all_gt_pitch_mean - all_gt_pitch_mean.mean()).mean():.4f}")
print(f"  Energy L1: {np.abs(all_gt_energy_mean - all_gt_energy_mean.mean()).mean():.4f}")

# Check: is pred std << GT std? (indicator of underfitting)
print(f"\nPred/GT std ratio:")
print(f"  Pitch: {all_pred_pitch.std() / all_gt_pitch_mean.std():.2f}x")
print(f"  Energy: {all_pred_energy.std() / all_gt_energy_mean.std():.2f}x")

# ============================================================================
# 3. Frame-level L1 breakdown by phoneme duration
# ============================================================================
print(f"\n{'='*70}")
print(f"FRAME-LEVEL LOSS CONTRIBUTION BY PHONEME DURATION")
print(f"{'='*70}")

# The training loss is frame-level: each phoneme with dur=N contributes N frames
# How much does each duration bucket contribute?
dur_bins = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 200)]
print(f"{'dur_range':>10} {'n_phon':>7} {'n_frames':>9} {'frame%':>7} {'pitch_L1':>9} {'energy_L1':>10}")
for lo, hi in dur_bins:
    mask = (all_durations >= lo) & (all_durations <= hi)
    if mask.sum() == 0:
        continue
    n_phon = mask.sum()
    n_frames = all_durations[mask].sum()
    pct = n_frames / all_durations.sum() * 100
    p_l1 = np.abs(all_pred_pitch[mask] - all_gt_pitch_mean[mask]).mean()
    e_l1 = np.abs(all_pred_energy[mask] - all_gt_energy_mean[mask]).mean()
    print(f"  {lo:>3}-{hi:<3}  {n_phon:>7} {n_frames:>9} {pct:>6.1f}% {p_l1:>9.4f} {e_l1:>10.4f}")

# ============================================================================
# 4. The real question: frame-level L1 as computed during training
# ============================================================================
print(f"\n{'='*70}")
print(f"SIMULATED FRAME-LEVEL LOSS (as training computes it)")
print(f"{'='*70}")

# In full_e2e training, pitch predictor output is expanded by PREDICTED durations
# and compared to frame-level GT. Let's simulate with GT durations for simplicity.
frame_level_pitch_errors = []
frame_level_energy_errors = []
for rec in sample:
    pid = rec["poem_id"]
    npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
    f0 = npz["f0"].astype(np.float32)
    energy = npz["energy"].astype(np.float32)
    durations = np.array(rec["durations"], dtype=int)
    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)

    f0_norm = np.where(f0 > 0,
        (np.log(np.maximum(f0, 1)) - f0_mean) / f0_std, 0.0).astype(np.float32)
    e_norm = ((energy - e_mean) / e_std).astype(np.float32)

    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        pred_pitch = model.pitch_predictor(x)[0].numpy()
        pred_energy = model.energy_predictor(x)[0].numpy()

    frame_idx = 0
    for i, dur in enumerate(durations):
        if dur == 0:
            continue
        for _ in range(dur):
            if frame_idx < len(f0_norm):
                frame_level_pitch_errors.append(abs(pred_pitch[i] - f0_norm[frame_idx]))
                frame_level_energy_errors.append(abs(pred_energy[i] - e_norm[frame_idx]))
            frame_idx += 1

print(f"\nFrame-level L1 (phoneme pred vs each frame GT):")
print(f"  Pitch:  {np.mean(frame_level_pitch_errors):.4f}")
print(f"  Energy: {np.mean(frame_level_energy_errors):.4f}")
print(f"  (Training log shows pitch~0.53, energy~0.76)")
print(f"  Phoneme-level L1 was: pitch={l1_phoneme:.4f}, energy={l1_e_phoneme:.4f}")

# ============================================================================
# 5. Weight analysis: are predictor weights tiny?
# ============================================================================
print(f"\n{'='*70}")
print(f"PREDICTOR WEIGHT ANALYSIS")
print(f"{'='*70}")

for name in ["pitch_predictor", "energy_predictor", "duration_predictor"]:
    pred = getattr(model, name)
    w = pred.linear.weight
    b = pred.linear.bias
    print(f"\n{name}:")
    print(f"  linear.weight: shape={list(w.shape)} abs_mean={w.abs().mean():.6f} std={w.std():.6f}")
    print(f"  linear.bias:   value={b.item():.4f}")
    conv1_w = pred.conv1.weight
    conv2_w = pred.conv2.weight
    print(f"  conv1.weight:  abs_mean={conv1_w.abs().mean():.6f} std={conv1_w.std():.6f}")
    print(f"  conv2.weight:  abs_mean={conv2_w.abs().mean():.6f} std={conv2_w.std():.6f}")

# ============================================================================
# 6. Check: does pitch_embed / energy_embed actually affect mel?
# ============================================================================
print(f"\n{'='*70}")
print(f"PITCH/ENERGY EMBEDDING WEIGHTS")
print(f"{'='*70}")

pw = model.pitch_embed.weight  # [d_model, 1]
ew = model.energy_embed.weight
print(f"pitch_embed.weight:  abs_mean={pw.abs().mean():.6f} std={pw.std():.6f}")
print(f"energy_embed.weight: abs_mean={ew.abs().mean():.6f} std={ew.std():.6f}")

# Compare to encoder output scale
print(f"\nFor reference, encoder output scale:")
dummy = torch.randn(1, 50, 256)
enc_scale = dummy.std().item()
print(f"  Random d_model=256 input std: {enc_scale:.4f}")
print(f"  pitch_embed contribution (per unit input): {pw.std().item():.6f}")
print(f"  Ratio pitch_embed/std: {pw.std().item() / enc_scale:.4f}")

print("\nDONE")
