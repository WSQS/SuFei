"""Separate silent vs speech mel frames to expose the real L1.

Hypothesis: 67% silent frames mask the true speech-frame error.
If silent-frame L1 ~= 0 and speech-frame L1 ~= 0.9, then the model
is essentially just learning to output "quiet" and failing on actual
speech content.

This script computes:
1. Per-frame L1 split by silent/speech (threshold from GT energy)
2. Frame energy distribution (GT vs pred)
3. What fraction of L1 mass comes from speech frames
4. Per-bin analysis: L1 by energy quintile
"""
import json, math, sys, numpy as np, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nar_fastspeech2 import FastSpeech2

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "paddle_distill_features"
STATS = json.load(open(ROOT / "data" / "train_300_norm_stats.json"))
mel_mean = np.array(STATS["mel_mean"], dtype=np.float32)
mel_std = np.array(STATS["mel_std"], dtype=np.float32)

ckpt = torch.load(str(ROOT / "checkpoints" / "FullE2E_step24000_slim.pt"),
                   map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()

import random
train = [json.loads(l) for l in open(ROOT / "data" / "train_300_manifest.jsonl", encoding="utf-8")]
random.seed(42)
sample = random.sample(train, 30)
sample.sort(key=lambda r: r["mel_len"])

# Energy threshold: a frame is "speech" if its raw mel energy > -5
ENERGY_THRESH = -5.0

# Aggregate bins: 5 energy quintiles
N_BINS = 5

all_frame_l1 = []      # (l1, is_speech) per frame, per sample
all_gt_energies = []   # frame energies from GT mel
all_pred_energies = [] # frame energies from pred mel
per_sample_stats = []

for idx, rec in enumerate(sample):
    pid = rec["poem_id"]
    npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
    mel_gt_raw = npz["mel"].astype(np.float32)
    mel_norm_gt = (mel_gt_raw - mel_mean) / mel_std
    dur_gt = np.array(rec["durations"], dtype=np.int32)
    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)

    # GT-var inference (perfect duration/pitch/energy)
    dur_gt_t = torch.tensor([rec["durations"]], dtype=torch.long)
    f0_gt = npz["f0"].astype(np.float32)
    e_gt = npz["energy"].astype(np.float32)
    f0_norm = np.where(f0_gt > 0,
        (np.log(np.maximum(f0_gt, 1)) - STATS["f0_mean"]) / STATS["f0_std"], 0.0).astype(np.float32)
    e_norm = ((e_gt - STATS["energy_mean"]) / STATS["energy_std"]).astype(np.float32)
    pitch_gt = torch.tensor(f0_norm).unsqueeze(0)
    energy_gt_t = torch.tensor(e_norm).unsqueeze(0)

    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        mi = model.length_regulator(x, dur_gt_t)
        T_out = mi.size(1)
        mi = mi + model.pitch_embed(pitch_gt[:, :T_out].unsqueeze(-1)) + \
                 model.energy_embed(energy_gt_t[:, :T_out].unsqueeze(-1))
        mi = model.pos_enc(mi)
        dec = mi
        for layer in model.decoder_layers:
            dec = layer(dec)
        mel_gtvar_norm = model.mel_linear(dec)[0].numpy()

    T = min(mel_gtvar_norm.shape[0], mel_norm_gt.shape[0])

    # Per-frame L1
    frame_l1 = np.abs(mel_gtvar_norm[:T] - mel_norm_gt[:T]).mean(axis=1)

    # Per-frame energy (from raw mel mean across freq bins)
    gt_frame_energy = mel_gt_raw[:T].mean(axis=1)
    pred_mel_raw = mel_gtvar_norm[:T] * mel_std + mel_mean
    pred_frame_energy = pred_mel_raw.mean(axis=1)

    is_speech = gt_frame_energy > ENERGY_THRESH

    # Aggregate
    for i in range(T):
        all_frame_l1.append((frame_l1[i], is_speech[i]))
    all_gt_energies.extend(gt_frame_energy.tolist())
    all_pred_energies.extend(pred_frame_energy.tolist())

    n_speech = is_speech.sum()
    n_silent = T - n_speech
    l1_speech = frame_l1[is_speech].mean() if n_speech > 0 else 0
    l1_silent = frame_l1[~is_speech].mean() if n_silent > 0 else 0
    l1_all = frame_l1.mean()

    per_sample_stats.append({
        "poem_id": pid,
        "n_frames": T,
        "n_speech": int(n_speech),
        "n_silent": int(n_silent),
        "speech_ratio": n_speech / T,
        "l1_speech": l1_speech,
        "l1_silent": l1_silent,
        "l1_all": l1_all,
        "l1_speech_contribution": (l1_speech * n_speech) / (l1_all * T),
        "text": rec["text"][:30],
    })

    if idx < 10 or idx % 10 == 0:
        print(f"[{idx+1}/30] {pid}: speech={n_speech}/{T} ({n_speech/T:.0%}) | "
              f"L1 speech={l1_speech:.3f} silent={l1_silent:.3f} all={l1_all:.3f} | "
              f"speech contributes {l1_speech*n_speech/(l1_all*T):.0%} of L1 mass")

# ============================================================================
# AGGREGATE ANALYSIS
# ============================================================================
all_l1 = np.array([x[0] for x in all_frame_l1])
all_speech = np.array([x[1] for x in all_frame_l1])

print(f"\n{'='*70}")
print(f"FRAME-LEVEL L1 ANALYSIS: SILENT vs SPEECH ({len(all_l1)} frames total)")
print(f"{'='*70}")

n_total = len(all_l1)
n_speech = all_speech.sum()
n_silent = n_total - n_speech

print(f"\nFrame count:")
print(f"  Speech: {n_speech} ({n_speech/n_total:.1%})")
print(f"  Silent: {n_silent} ({n_silent/n_total:.1%})")

l1_speech = all_l1[all_speech].mean()
l1_silent = all_l1[~all_speech].mean()
l1_all = all_l1.mean()

print(f"\nMel L1 (normalized):")
print(f"  All frames:    {l1_all:.4f}")
print(f"  Speech frames: {l1_speech:.4f}  ({l1_speech/l1_all:.1f}x overall)")
print(f"  Silent frames: {l1_silent:.4f}  ({l1_silent/l1_all:.1f}x overall)")

# L1 mass contribution
mass_speech = (l1_speech * n_speech) / (l1_all * n_total)
mass_silent = (l1_silent * n_silent) / (l1_all * n_total)
print(f"\nL1 error mass:")
print(f"  From speech frames: {mass_speech:.1%}")
print(f"  From silent frames: {mass_silent:.1%}")

# Per-sample summary
print(f"\n{'='*70}")
print(f"PER-SAMPLE SUMMARY (sorted by speech L1)")
print(f"{'='*70}")
print(f"{'poem_id':<16} {'spk%':>5} {'L1_spk':>8} {'L1_sil':>8} {'L1_all':>8} {'mass%':>6}")
for r in sorted(per_sample_stats, key=lambda x: -x["l1_speech"]):
    print(f"{r['poem_id']:<16} {r['speech_ratio']:>4.0%} {r['l1_speech']:>8.3f} "
          f"{r['l1_silent']:>8.3f} {r['l1_all']:>8.3f} {r['l1_speech_contribution']:>5.0%}")

# ============================================================================
# ENERGY QUINTILE ANALYSIS
# ============================================================================
print(f"\n{'='*70}")
print(f"L1 BY GT ENERGY QUINTILE")
print(f"{'='*70}")

gt_energies = np.array(all_gt_energies)
pred_energies = np.array(all_pred_energies)
frame_l1s = all_l1

# Bin frames by GT energy into quintiles
quintiles = np.quantile(gt_energies, [0.2, 0.4, 0.6, 0.8])
bin_idx = np.digitize(gt_energies, quintiles)  # 0..4

print(f"{'quintile':>8} {'energy range':>20} {'n_frames':>9} {'L1_mean':>8} {'pred_e':>8} {'gt_e':>8}")
for b in range(5):
    mask = bin_idx == b
    if mask.sum() == 0:
        continue
    e_lo = gt_energies[mask].min()
    e_hi = gt_energies[mask].max()
    l1_q = frame_l1s[mask].mean()
    pred_e = pred_energies[mask].mean()
    gt_e = gt_energies[mask].mean()
    label = " (silent)" if b == 0 else (" (speech)" if b == 4 else "")
    print(f"  Q{b+1}     [{e_lo:>7.2f}, {e_hi:>7.2f}] {mask.sum():>8} {l1_q:>8.3f} "
          f"{pred_e:>8.2f} {gt_e:>8.2f}{label}")

# ============================================================================
# WHAT DOES THE MODEL ACTUALLY OUTPUT ON SPEECH FRAMES?
# ============================================================================
print(f"\n{'='*70}")
print(f"MODEL OUTPUT ON SPEECH FRAMES (worst 5 samples)")
print(f"{'='*70}")

# For worst samples, show the model's mel on speech frames
# Compare a few mel bins to see if it's outputting noise vs structured
for r in sorted(per_sample_stats, key=lambda x: -x["l1_speech"])[:3]:
    pid = r["poem_id"]
    rec = next(x for x in sample if x["poem_id"] == pid)
    npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
    mel_gt_raw = npz["mel"].astype(np.float32)
    mel_norm_gt = (mel_gt_raw - mel_mean) / mel_std

    # Re-run GT-var
    dur_gt_t = torch.tensor([rec["durations"]], dtype=torch.long)
    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
    f0_gt = npz["f0"].astype(np.float32)
    e_gt = npz["energy"].astype(np.float32)
    f0_norm = np.where(f0_gt > 0,
        (np.log(np.maximum(f0_gt, 1)) - STATS["f0_mean"]) / STATS["f0_std"], 0.0).astype(np.float32)
    e_norm = ((e_gt - STATS["energy_mean"]) / STATS["energy_std"]).astype(np.float32)

    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        mi = model.length_regulator(x, dur_gt_t)
        T_out = mi.size(1)
        pitch_gt = torch.tensor(f0_norm).unsqueeze(0)
        energy_gt_t = torch.tensor(e_norm).unsqueeze(0)
        mi = mi + model.pitch_embed(pitch_gt[:, :T_out].unsqueeze(-1)) + \
                 model.energy_embed(energy_gt_t[:, :T_out].unsqueeze(-1))
        mi = model.pos_enc(mi)
        dec = mi
        for layer in model.decoder_layers:
            dec = layer(dec)
        mel_pred = model.mel_linear(dec)[0].numpy()

    T = min(mel_pred.shape[0], mel_norm_gt.shape[0])
    gt_energy = mel_gt_raw[:T].mean(axis=1)
    is_speech = gt_energy > ENERGY_THRESH

    # Show first 10 speech frames: GT vs pred for first 5 mel bins
    speech_idx = np.where(is_speech)[0]
    if len(speech_idx) > 0:
        print(f"\n  {pid} ({r['text']}): L1_speech={r['l1_speech']:.3f}")
        print(f"  First 5 speech frames (mel bins 0-4):")
        print(f"    {'frame':>5} {'GT[0:5]':>40} {'Pred[0:5]':>40}")
        for fi in speech_idx[:5]:
            gt_vals = mel_norm_gt[fi, :5]
            pr_vals = mel_pred[fi, :5]
            print(f"    {fi:>5} [{', '.join(f'{v:>6.2f}' for v in gt_vals)}] "
                  f"[{', '.join(f'{v:>6.2f}' for v in pr_vals)}]")

        # Spectral contrast: std across mel bins (should be structured, not flat)
        gt_speech = mel_norm_gt[speech_idx]
        pred_speech = mel_pred[speech_idx]
        gt_spec_std = gt_speech.std(axis=1).mean()
        pred_spec_std = pred_speech.std(axis=1).mean()
        print(f"  Spectral contrast (std across freq bins):")
        print(f"    GT:   {gt_spec_std:.3f}")
        print(f"    Pred: {pred_spec_std:.3f}  ({pred_spec_std/gt_spec_std:.1%} of GT)")
        print(f"  -> {'Model output is FLAT (spectrally blurred)' if pred_spec_std < gt_spec_std * 0.7 else 'Model output has spectral structure'}")

print("\nDONE")
