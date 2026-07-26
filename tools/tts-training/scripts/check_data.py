"""Quick data inspection for NAR training manifest build."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
feat_dir = ROOT / "data" / "nar_features"
dur_file = ROOT / "data" / "duration_labels.json"

with open(dur_file, encoding="utf-8") as f:
    dur = json.load(f)

s = dur["samples"][0]
print(f"Poem: {s['poem_id']} - {s['title']}")
print(f"  text: {s['full_text'][:40]}")
print(f"  char_durations count: {len(s['char_durations'])}")
print(f"  mel_durations len: {len(s['mel_durations'])}")
print(f"  total_mel_frames: {s['total_mel_frames']}")

npz = np.load(str(feat_dir / f"{s['poem_id']}.npz"))
print(f"  mel: {npz['mel'].shape}")
print(f"  f0: {npz['f0'].shape}")
print(f"  energy: {npz['energy'].shape}")
print(f"  durations: {npz['durations'].shape}, sum={npz['durations'].sum()}")
print(f"  mel_len: {npz['mel'].shape[0]}")
print(f"  gap: {npz['mel'].shape[0] - npz['durations'].sum()}")

chars = [(i, cd) for i, cd in enumerate(s["char_durations"]) if cd["type"] == "char"]
puncts = [(i, cd) for i, cd in enumerate(s["char_durations"]) if cd["type"] == "punct"]
print(f"  chars: {len(chars)}, puncts: {len(puncts)}")
print(f"  durations for puncts: {[int(npz['durations'][i]) for i, _ in puncts]}")

# Check distribution of gaps across all samples
print("\n--- Gap distribution across all samples ---")
gaps = []
for s in dur["samples"]:
    npz_path = feat_dir / f"{s['poem_id']}.npz"
    if not npz_path.exists():
        continue
    d = np.load(str(npz_path))
    mel_len = d["mel"].shape[0]
    dur_sum = int(d["durations"].sum())
    gap = mel_len - dur_sum
    gaps.append(gap)

gaps = np.array(gaps)
print(f"  N={len(gaps)}")
print(f"  gap min={gaps.min()}, max={gaps.max()}, mean={gaps.mean():.1f}, median={np.median(gaps):.1f}")
print(f"  gap > 0 count: {(gaps > 0).sum()} (punctuation silence not allocated)")
print(f"  gap == 0 count: {(gaps == 0).sum()}")
print(f"  gap < 0 count: {(gaps < 0).sum()}")
