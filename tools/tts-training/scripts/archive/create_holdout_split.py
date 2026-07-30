"""Create stratified holdout split from 191 training poems.

Produces:
  data/train_171_manifest.jsonl   — 171 poems for training
  data/holdout_20_manifest.jsonl  — 20 poems for evaluation, same distribution
  data/external_unseen_6_manifest.jsonl — 6 external unseen poems

Selection criteria:
  - Stratified by mel_len (short/mid/long)
  - Text hash deduplication
  - Cross-set disjoint assertions
"""
import json
import hashlib
import re
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")

def normalize_body(text):
    """Remove punctuation, title/author prefix, whitespace."""
    # Remove punctuation and whitespace
    t = re.sub(r"[，。！？；：、·\s\n]", "", text)
    return t

def text_hash(text):
    return hashlib.sha256(normalize_body(text).encode("utf-8")).hexdigest()

# Load all 191 poems
all_poems = []
with open(ROOT / "data/paddle_distill_manifest.jsonl", encoding="utf-8") as f:
    for line in f:
        all_poems.append(json.loads(line))

print(f"Total poems: {len(all_poems)}")

# Compute mel_len tertiles
mel_lens = [p["mel_len"] for p in all_poems]
p33 = np.percentile(mel_lens, 33)
p67 = np.percentile(mel_lens, 67)
print(f"mel_len tertiles: P33={p33:.0f}, P67={p67:.0f}")

# Classify into short/mid/long
for p in all_poems:
    if p["mel_len"] <= p33:
        p["_len_bin"] = "short"
    elif p["mel_len"] <= p67:
        p["_len_bin"] = "mid"
    else:
        p["_len_bin"] = "long"

bin_counts = {"short": 0, "mid": 0, "long": 0}
for p in all_poems:
    bin_counts[p["_len_bin"]] += 1
print(f"Length bins: {bin_counts}")

# Target: 7 short, 7 mid, 6 long
targets = {"short": 7, "mid": 7, "long": 6}
np.random.seed(42)

holdout = []
train = list(all_poems)  # copy; will remove holdout

for bin_name in ["short", "mid", "long"]:
    bin_poems = [p for p in all_poems if p["_len_bin"] == bin_name]
    np.random.shuffle(bin_poems)
    selected = bin_poems[:targets[bin_name]]
    holdout.extend(selected)
    print(f"  {bin_name}: selected {len(selected)} from {len(bin_poems)}")

# Remove holdout from train
holdout_ids = set(p["poem_id"] for p in holdout)
train = [p for p in all_poems if p["poem_id"] not in holdout_ids]

print(f"\nTrain: {len(train)}, Holdout: {len(holdout)}")

# Compute hashes and assert disjoint
train_hashes = set(text_hash(p["text"]) for p in train)
holdout_hashes = set(text_hash(p["text"]) for p in holdout)

# Load external unseen
external = []
with open(ROOT / "data/unseen_manifest.jsonl", encoding="utf-8") as f:
    for line in f:
        external.append(json.loads(line))
external_hashes = set(text_hash(p["text"]) for p in external)

assert train_hashes.isdisjoint(holdout_hashes), "TRAIN-HOLDOUT OVERLAP!"
assert train_hashes.isdisjoint(external_hashes), "TRAIN-EXTERNAL OVERLAP!"
assert holdout_hashes.isdisjoint(external_hashes), "HOLDOUT-EXTERNAL OVERLAP!"
print("Hash disjoint assertions: ALL PASS")

# Write manifests (strip internal fields)
def clean(poems):
    cleaned = []
    for p in poems:
        c = {k: v for k, v in p.items() if not k.startswith("_")}
        cleaned.append(c)
    return cleaned

train_out = ROOT / "data/train_171_manifest.jsonl"
holdout_out = ROOT / "data/holdout_20_manifest.jsonl"
external_out = ROOT / "data/external_unseen_6_manifest.jsonl"

with open(train_out, "w", encoding="utf-8") as f:
    for p in clean(train):
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

with open(holdout_out, "w", encoding="utf-8") as f:
    for p in clean(holdout):
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

with open(external_out, "w", encoding="utf-8") as f:
    for p in clean(external):
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

# Also save split metadata
split_meta = {
    "train_171": {
        "count": len(train),
        "file": "data/train_171_manifest.jsonl",
        "mel_len_mean": float(np.mean([p["mel_len"] for p in train])),
        "mel_len_median": float(np.median([p["mel_len"] for p in train])),
    },
    "holdout_20": {
        "count": len(holdout),
        "file": "data/holdout_20_manifest.jsonl",
        "mel_len_mean": float(np.mean([p["mel_len"] for p in holdout])),
        "mel_len_median": float(np.median([p["mel_len"] for p in holdout])),
        "poems": [{"poem_id": p["poem_id"], "mel_len": p["mel_len"], 
                    "len_bin": p["_len_bin"], "text": p["text"][:40]}
                   for p in sorted(holdout, key=lambda x: x["mel_len"])],
    },
    "external_unseen_6": {
        "count": len(external),
        "file": "data/external_unseen_6_manifest.jsonl",
    },
    "seed": 42,
}

with open(ROOT / "data/split_meta.json", "w", encoding="utf-8") as f:
    json.dump(split_meta, f, ensure_ascii=False, indent=2)

# Print holdout details
print(f"\n{'='*70}")
print("HOLDOUT 20 (frozen)")
print(f"{'='*70}")
print(f"{'poem_id':<18} {'mel_len':>7} {'bin':>6} {'text'}")
print("-" * 70)
for p in sorted(holdout, key=lambda x: x["mel_len"]):
    print(f"{p['poem_id']:<18} {p['mel_len']:>7} {p['_len_bin']:>6} {p['text'][:40]}")

print(f"\nTrain mel_len:  mean={split_meta['train_171']['mel_len_mean']:.0f}  median={split_meta['train_171']['mel_len_median']:.0f}")
print(f"Holdout mel_len: mean={split_meta['holdout_20']['mel_len_mean']:.0f}  median={split_meta['holdout_20']['mel_len_median']:.0f}")

print(f"\nFiles written:")
print(f"  {train_out}")
print(f"  {holdout_out}")
print(f"  {external_out}")
print(f"  {ROOT / 'data/split_meta.json'}")
