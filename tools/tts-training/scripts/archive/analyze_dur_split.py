"""Analyze duration distribution: punctuation vs character durations."""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = [json.loads(l) for l in open(ROOT / "data" / "train_manifest.jsonl", encoding="utf-8")]

total_all = 0
total_char = 0
total_punct = 0

for rec in manifest:
    durs = np.array(rec["durations"])
    punct_mask = durs > 50  # punctuation pauses
    total_all += durs.sum()
    total_char += durs[~punct_mask].sum()
    total_punct += durs[punct_mask].sum()

print(f"Total frames: {total_all}")
print(f"Character frames: {total_char} ({total_char/total_all:.1%})")
print(f"Punctuation frames: {total_punct} ({total_punct/total_all:.1%})")
print(f"\nIf duration predictor correctly predicts character durations")
print(f"but can't predict punctuation pauses (>50 frames):")
print(f"  Expected ratio = {total_char/total_all:.1%}")
print(f"  This matches the ~30% ceiling!")
