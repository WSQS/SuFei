"""Compute normalization stats for train_300."""
import json, os, sys
import numpy as np

ROOT = r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"

manifest = os.path.join(ROOT, "data", "train_300_manifest.jsonl")
feat_dir = os.path.join(ROOT, "data", "paddle_distill_features")
stats_path = os.path.join(ROOT, "data", "train_300_norm_stats.json")

# Inline TTSDataset loading
records = []
with open(manifest, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["mel_len"] <= 2000:
            r["_npz"] = os.path.join(feat_dir, "%s.npz" % r["poem_id"])
            records.append(r)
print("Records: %d" % len(records))

all_mel = []
all_f0_voiced = []
all_energy = []

for r in records:
    npz = np.load(r["_npz"])
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
f0_std = float(f0_cat.std()) + 1e-8

energy_cat = np.concatenate(all_energy)
energy_mean = float(energy_cat.mean())
energy_std = float(energy_cat.std()) + 1e-8

stats = {
    "mel_mean": mel_mean.tolist(),
    "mel_std": mel_std.tolist(),
    "f0_mean": f0_mean,
    "f0_std": f0_std,
    "energy_mean": energy_mean,
    "energy_std": energy_std,
}

with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

print("Saved: %s" % stats_path)
print("mel_mean range: [%.2f, %.2f]" % (mel_mean.min(), mel_mean.max()))
print("f0_mean=%.2f f0_std=%.2f" % (f0_mean, f0_std))
print("energy_mean=%.2f energy_std=%.2f" % (energy_mean, energy_std))

# Compare with train_171 stats
t171 = json.load(open(os.path.join(ROOT, "data", "train_171_norm_stats.json")))
diff = np.abs(mel_mean - np.array(t171["mel_mean"], dtype=np.float32))
print("mel_mean max diff vs train_171: %.4f" % diff.max())
