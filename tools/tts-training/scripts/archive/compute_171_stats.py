"""Compute norm stats from train_171 only (holdout excluded)."""
import json, numpy as np
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")

all_mel = []
all_f0_voiced = []
all_energy = []

with open(ROOT / "data/train_171_manifest.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        npz = np.load(str(ROOT / "data/paddle_distill_features" / f"{rec['poem_id']}.npz"))
        mel = npz["mel"].astype(np.float32)
        f0 = npz["f0"].astype(np.float32)
        energy = npz["energy"].astype(np.float32)
        
        all_mel.append(mel)
        voiced = f0[f0 > 0]
        if len(voiced) > 0:
            all_f0_voiced.append(np.log(voiced))
        all_energy.append(energy)

mel_cat = np.concatenate(all_mel, axis=0)
mel_mean = mel_cat.mean(axis=0)
mel_std = mel_cat.std(axis=0) + 1e-8

f0_cat = np.concatenate(all_f0_voiced)
f0_mean = float(f0_cat.mean())
f0_std = float(f0_cat.std() + 1e-8)

energy_cat = np.concatenate(all_energy)
energy_mean = float(energy_cat.mean())
energy_std = float(energy_cat.std() + 1e-8)

stats = {
    "mel_mean": mel_mean.tolist(),
    "mel_std": mel_std.tolist(),
    "f0_mean": f0_mean,
    "f0_std": f0_std,
    "energy_mean": energy_mean,
    "energy_std": energy_std,
}

out_path = ROOT / "data/train_171_norm_stats.json"
with open(out_path, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Saved {out_path}")
print(f"  mel_mean range: [{mel_mean.min():.2f}, {mel_mean.max():.2f}]")
print(f"  mel_std range:  [{mel_std.min():.4f}, {mel_std.max():.4f}]")
print(f"  f0_mean={f0_mean:.2f}, f0_std={f0_std:.2f}")
print(f"  energy_mean={energy_mean:.2f}, energy_std={energy_std:.2f}")

# Compare with 191 stats
with open(ROOT / "data/paddle_distill_norm_stats.json") as f:
    stats191 = json.load(f)
mel_mean_191 = np.array(stats191["mel_mean"])
mel_std_191 = np.array(stats191["mel_std"])
print(f"\nDiff from 191 stats:")
print(f"  mel_mean max abs diff: {np.abs(mel_mean - mel_mean_191).max():.4f}")
print(f"  mel_std max abs diff:  {np.abs(mel_std - mel_std_191).max():.4f}")
print(f"  f0_mean diff:          {abs(f0_mean - stats191['f0_mean']):.4f}")
