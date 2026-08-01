"""One-off: asr_cer distribution of the m3 train manifest (numbers only)."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cers = []
for line in open(ROOT / "data" / "m3_train_manifest.jsonl", encoding="utf-8"):
    cers.append(json.loads(line)["asr_cer"])
print("n:", len(cers))
print("mean:", round(statistics.mean(cers), 4))
print("median:", round(statistics.median(cers), 4))
print("zero:", sum(1 for c in cers if c == 0))
print("0-5pct:", sum(1 for c in cers if 0 < c <= 0.05))
print("over5pct:", sum(1 for c in cers if c > 0.05))
print("top10:", sorted((round(c, 3) for c in cers), reverse=True)[:10])
