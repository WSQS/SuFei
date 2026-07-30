"""Diagnose training loss — check mel normalization and model output scale."""
import json
import sys
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nar_fastspeech2 import FastSpeech2
from nar_train import TTSDataset, collate_fn, compute_stats, length_regulate_batch, masked_l1_loss

MANIFEST = ROOT / "data" / "train_manifest.jsonl"
feature_dir = ROOT / "data" / "nar_features"

dataset = TTSDataset(MANIFEST, feature_dir, max_mel_len=2000)

# Load stats
with open(ROOT / "data" / "norm_stats.json") as f:
    raw = json.load(f)

mel_mean = np.array(raw["mel_mean"], dtype=np.float32)
mel_std = np.array(raw["mel_std"], dtype=np.float32)

print("=== Mel stats ===")
print(f"  mel_mean: shape={mel_mean.shape}, range=[{mel_mean.min():.2f}, {mel_mean.max():.2f}]")
print(f"  mel_std:  shape={mel_std.shape}, range=[{mel_std.min():.4f}, {mel_std.max():.4f}]")

# Check raw mel distribution
item = dataset[0]
mel_raw = item["mel"]
print(f"\n=== Raw mel (sample 0) ===")
print(f"  shape: {mel_raw.shape}")
print(f"  range: [{mel_raw.min():.2f}, {mel_raw.max():.2f}]")
print(f"  mean:  {mel_raw.mean():.2f}")
print(f"  std:   {mel_raw.std():.2f}")

mel_normed = (mel_raw - mel_mean) / mel_std
print(f"\n=== Normalized mel (sample 0) ===")
print(f"  range: [{mel_normed.min():.2f}, {mel_normed.max():.2f}]")
print(f"  mean:  {mel_normed.mean():.2f}")
print(f"  std:   {mel_normed.std():.2f}")

# Check model output scale
device = "cuda"
model = FastSpeech2(vocab_size=268, dropout=0.0).to(device)

# Single sample forward
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
batch = next(iter(loader))

phoneme_ids = batch["phoneme_ids"].to(device)
durations_gt = batch["durations"].to(device)
mel_gt = batch["mel"].to(device)
phone_mask = batch["phone_mask"].to(device)

mel_mean_t = torch.tensor(mel_mean, device=device)
mel_std_t = torch.tensor(mel_std, device=device)
mel_norm_gt = (mel_gt - mel_mean_t) / mel_std_t

print(f"\n=== Batch ===")
print(f"  phoneme_ids: {phoneme_ids.shape}")
print(f"  durations_gt: {durations_gt.shape}")
print(f"  mel_gt: {mel_gt.shape}")
print(f"  mel_norm_gt: mean={mel_norm_gt.mean():.2f}, std={mel_norm_gt.std():.2f}")
print(f"  dur sums per sample: {durations_gt.sum(dim=1).tolist()}")
print(f"  mel_lens: {batch['mel_lens'].tolist()}")

with torch.no_grad():
    mel_pred, log_dur_pred, pitch_pred, energy_pred = model(
        phoneme_ids, durations=durations_gt, phone_mask=phone_mask,
    )

print(f"\n=== Model output (untrained) ===")
print(f"  mel_pred: {mel_pred.shape}")
print(f"  range: [{mel_pred.min():.2f}, {mel_pred.max():.2f}]")
print(f"  mean:  {mel_pred.mean():.2f}")
print(f"  std:   {mel_pred.std():.2f}")

# L1 loss
T_min = min(mel_pred.size(1), mel_norm_gt.size(1))
mel_mask = batch["mel_mask"].to(device)
l1 = masked_l1_loss(mel_pred[:, :T_min], mel_norm_gt[:, :T_min], mel_mask[:, :T_min])
print(f"\n  L1 loss (untrained): {l1.item():.4f}")

# Check if mel_pred is NaN or Inf
print(f"\n  NaN count: {torch.isnan(mel_pred).sum().item()}")
print(f"  Inf count: {torch.isinf(mel_pred).sum().item()}")
