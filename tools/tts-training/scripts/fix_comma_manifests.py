"""Fix phoneme_ids in all manifests after comma bug fix.

Regenerates phoneme_ids using corrected G2P, preserving everything else
(durations, mel_len, text, asr_cer, mel_path, n_phonemes).
Verifies: len(new_ids) == len(old_ids) for every record.
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
sys.path.insert(0, str(ROOT / "scripts"))

from generate_paddlespeech_distillation_data import (
    text_to_phonemes,
    phones_to_ids,
    load_phone_id_map,
)

phone_map = load_phone_id_map()

MANIFESTS = [
    "data/paddle_distill_manifest.jsonl",
    "data/train_171_manifest.jsonl",
    "data/holdout_20_manifest.jsonl",
    "data/external_unseen_6_manifest.jsonl",
    "data/unseen_manifest.jsonl",
]

total_fixed = 0
total_ok = 0
total_mismatch = 0

for mf in MANIFESTS:
    path = ROOT / mf
    if not path.exists():
        print("SKIP (not found): %s" % mf)
        continue

    records = []
    fixed = 0
    mismatches = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = r["text"]
            old_ids = r["phoneme_ids"]

            phones = text_to_phonemes(text)
            new_ids, unmapped = phones_to_ids(phones, phone_map)

            if len(new_ids) != len(old_ids):
                print("  LENGTH MISMATCH %s: old=%d new=%d" % (
                    r["poem_id"], len(old_ids), len(new_ids)))
                mismatches += 1
                records.append(r)
                continue

            changed = sum(1 for a, b in zip(old_ids, new_ids) if a != b)
            if changed > 0:
                fixed += 1

            r["phoneme_ids"] = new_ids
            records.append(r)

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    total_fixed += fixed
    total_ok += (n - mismatches)
    total_mismatch += mismatches
    print("FIXED %-45s: %d records, %d changed, %d mismatches" % (
        mf, n, fixed, mismatches))

print()
print("Total: %d fixed, %d ok, %d mismatches" % (total_fixed, total_ok, total_mismatch))
