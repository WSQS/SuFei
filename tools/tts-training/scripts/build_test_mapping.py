"""Build poem_id → teacher_wav mapping by matching title from test_set to train_raw.jsonl."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load test set
with open(ROOT / "data" / "test_set_v1.json", encoding="utf-8") as f:
    test_set = json.load(f)

# Load train_raw to get hash → title mapping
train = []
with open(ROOT / "data" / "train_raw.jsonl", encoding="utf-8") as f:
    for line in f:
        train.append(json.loads(line.strip()))

# Build title → wav mapping from train data
title_to_wav = {}
for r in train:
    audio = r["audio"]  # e.g. "./audio/c35a60c1a8e2.wav"
    wav_name = Path(audio).stem  # c35a60c1a8e2
    text = r["text"]
    title = text.split("，")[0]
    title_to_wav[title] = wav_name

# Match test set
mapping = {}
for s in test_set["samples"]:
    title = s["title"]
    wav_hash = title_to_wav.get(title)
    if wav_hash:
        mapping[s["poem_id"]] = f"{wav_hash}.wav"
        s["teacher_wav"] = f"{wav_hash}.wav"
    else:
        print(f"  NOT FOUND: {s['poem_id']} ({title})")

# Update test set with teacher_wav field
with open(ROOT / "data" / "test_set_v1.json", "w", encoding="utf-8") as f:
    json.dump(test_set, f, ensure_ascii=False, indent=2)

print(f"\nMapped {len(mapping)}/{len(test_set['samples'])} samples")
for pid, wav in sorted(mapping.items()):
    print(f"  {pid:40s} -> {wav}")
