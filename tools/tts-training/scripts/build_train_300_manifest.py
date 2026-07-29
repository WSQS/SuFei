"""Build MFA corpus for 129 new poems, run MFA, then build manifest entries."""
import json
import os
import shutil
import subprocess
import sys

ROOT = r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from pathlib import Path
from generate_paddlespeech_distillation_data import (
    text_to_phonemes, phones_to_ids, load_phone_id_map,
    parse_textgrid_intervals, build_durations,
    FS2_DIR, FRAME_RATE,
)

DICTIONARY = os.path.join(ROOT, "data", "mfa_char_dictionary.txt")
MFA_CORPUS = os.path.join(ROOT, "data", "mfa_new129_corpus")
MFA_OUTPUT = os.path.join(ROOT, "data", "mfa_new129_aligned")
SRC_WAV_DIR = os.path.join(ROOT, "data", "paddle_mfa_corpus")
FEAT_DIR = os.path.join(ROOT, "data", "paddle_distill_features")

# Load selected 129 records
merged = json.load(open(os.path.join(ROOT, "data", "new_129_stage1_merged.json"), encoding="utf-8"))
pass_ids = [l.strip() for l in open(os.path.join(ROOT, "data", "new_129_pass_ids.txt"))]
selected = [x for x in merged if x["poem_id"] in set(pass_ids)]
print("Selected: %d poems" % len(selected))

# Build MFA corpus
if os.path.exists(MFA_CORPUS):
    shutil.rmtree(MFA_CORPUS)
os.makedirs(MFA_CORPUS, exist_ok=True)

if os.path.exists(MFA_OUTPUT):
    shutil.rmtree(MFA_OUTPUT)
os.makedirs(MFA_OUTPUT, exist_ok=True)

for entry in selected:
    pid = entry["poem_id"]
    for ext in [".wav", ".lab"]:
        src = os.path.join(SRC_WAV_DIR, "%s%s" % (pid, ext))
        if os.path.exists(src):
            shutil.copy2(src, MFA_CORPUS)

n_files = len([f for f in os.listdir(MFA_CORPUS) if f.endswith(".wav")])
print("MFA corpus: %d wavs" % n_files)

# Run MFA (skip if already done)
tg_existing = len([f for f in os.listdir(MFA_OUTPUT) if f.endswith(".TextGrid")]) if os.path.exists(MFA_OUTPUT) else 0
if tg_existing >= n_files:
    print("Skipping MFA (already have %d TextGrids)" % tg_existing)
else:
    cmd = [
        r"C:\Users\wsqsy\.conda\envs\mfa\Scripts\mfa.exe",
        "align",
        MFA_CORPUS,
        DICTIONARY,
        "mandarin_mfa",
        MFA_OUTPUT,
        "--overwrite",
        "--clean",
        "--num_jobs", "4",
        "--beam", "100",
        "--retry_beam", "400",
    ]
    print("Running MFA...")
    env = os.environ.copy()
    env["PATH"] = r"C:\Users\wsqsy\.conda\envs\mfa\Library\bin;" + env.get("PATH", "")
    result = subprocess.run(cmd, capture_output=False, timeout=3600, env=env)

tg_count = len([f for f in os.listdir(MFA_OUTPUT) if f.endswith(".TextGrid")])
print("MFA done: %d TextGrids" % tg_count)

# Build manifest entries
phone_map = load_phone_id_map()
records = []
mfa_fail = 0
dur_mismatch = 0

for entry in selected:
    pid = entry["poem_id"]
    text = entry["text"]
    mel_len = entry["mel_len"]

    tg_path = os.path.join(MFA_OUTPUT, "%s.TextGrid" % pid)
    if not os.path.exists(tg_path):
        print("  SKIP %s: no TextGrid" % pid)
        mfa_fail += 1
        continue

    tg_intervals = parse_textgrid_intervals(Path(tg_path))
    phones = text_to_phonemes(text)
    durations = build_durations(phones, tg_intervals)

    dur_sum = sum(durations)
    if dur_sum != mel_len:
        diff = mel_len - dur_sum
        max_idx = max(range(len(durations)), key=lambda x: durations[x])
        durations[max_idx] = max(durations[max_idx] + diff, 1)

    if sum(durations) != mel_len:
        print("  SKIP %s: dur mismatch %d vs %d" % (pid, sum(durations), mel_len))
        dur_mismatch += 1
        continue

    records.append({
        "poem_id": pid,
        "phoneme_ids": entry["phoneme_ids"],
        "durations": durations,
        "mel_path": "data/paddle_distill_features/%s.npz" % pid,
        "mel_len": mel_len,
        "n_phonemes": len(entry["phoneme_ids"]),
        "text": text,
        "asr_cer": entry["asr_cer"],
    })

# Save new 129 manifest
out_path = os.path.join(ROOT, "data", "new_129_distilled_manifest.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\nManifest built: %d records (mfa_fail=%d, dur_mismatch=%d)" % (
    len(records), mfa_fail, dur_mismatch))

# Verify invariant
bad = sum(1 for r in records if sum(r["durations"]) != r["mel_len"])
print("Duration invariant violations: %d/%d" % (bad, len(records)))

# Now build train_300 = train_171 + new_129
train171 = [json.loads(l) for l in open(
    os.path.join(ROOT, "data", "train_171_manifest.jsonl"), encoding="utf-8")]
train300 = train171 + records

train300_path = os.path.join(ROOT, "data", "train_300_manifest.jsonl")
with open(train300_path, "w", encoding="utf-8") as f:
    for r in train300:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Verify nesting
t171_ids = set(r["poem_id"] for r in train171)
t300_ids = set(r["poem_id"] for r in train300)
print("\ntrain_300: %d poems (train_171=%d ⊂ train_300=%s)" % (
    len(train300), len(t171_ids), t171_ids.issubset(t300_ids)))

# Check no overlap with holdout/external
holdout = [json.loads(l) for l in open(
    os.path.join(ROOT, "data", "holdout_20_manifest.jsonl"), encoding="utf-8")]
external = [json.loads(l) for l in open(
    os.path.join(ROOT, "data", "external_unseen_6_manifest.jsonl"), encoding="utf-8")]
holdout_ids = set(r["poem_id"] for r in holdout)
external_ids = set(r["poem_id"] for r in external)
overlap_h = t300_ids & holdout_ids
overlap_e = t300_ids & external_ids
print("Overlap with holdout: %d" % len(overlap_h))
print("Overlap with external: %d" % len(overlap_e))

print("\nSaved: %s" % train300_path)
