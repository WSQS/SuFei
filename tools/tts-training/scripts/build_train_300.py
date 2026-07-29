"""Build train_300: select 129 new Tang poems and prepare distillation source manifest.

Selection criteria:
- 20-60 content chars
- Not in train_171, holdout_20, or external_unseen_6
- Prefer shorter poems for sequence diversity
- Deduplicate by normalized text
- Fixed seed for reproducibility
"""
import json
import os
import random
import re
import hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def normalize(text):
    return re.sub(r"[，。？！；：、·\s\n]", "", text)

def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# Load all reserved texts
reserved_norms = set()
for mf in ["train_171_manifest.jsonl", "holdout_20_manifest.jsonl",
           "external_unseen_6_manifest.jsonl"]:
    path = os.path.join(ROOT, "data", mf)
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            reserved_norms.add(normalize(r["text"]))

# Load existing source poem_ids to avoid re-selecting
source_path = os.path.join(ROOT, "data", "train_manifest.jsonl")
existing_ids = set()
with open(source_path, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        existing_ids.add(r["poem_id"])
        reserved_norms.add(normalize(r["text"]))

# Load Tang dynasty poems from assets
assets = os.path.join(
    os.path.dirname(os.path.dirname(ROOT)),  # SuFei/
    "app", "src", "main", "assets",
)

candidates = []
seen_norms = set()
for i in range(5):
    path = os.path.join(assets, "poems_%d.jsonl" % i)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("dynasty") != "唐代":
                continue
            content = r.get("content", "")
            chars = [c for c in content if "\u4e00" <= c <= "\u9fff"]
            n_chars = len(chars)
            if not (20 <= n_chars <= 56):
                continue

            title = r["title"].strip()
            author = r["author"].strip()
            content_clean = content.replace("\n", "")

            full_text = "%s，%s·%s。%s" % (title, "唐代", author, content_clean)
            norm = normalize(full_text)

            if norm in reserved_norms or norm in seen_norms:
                continue
            seen_norms.add(norm)

            # Build poem_id from hash
            pid = "poem_%04d" % (300 + len(candidates))

            candidates.append({
                "poem_id": pid,
                "title": title,
                "author": author,
                "content": content_clean,
                "text": full_text,
                "n_chars": n_chars,
                "text_hash": text_hash(full_text),
            })

print("Total candidates: %d" % len(candidates))

# Sort by length (shorter first for more independent sequences),
# then shuffle within length buckets for diversity
random.seed(42)
buckets = {}
for c in candidates:
    key = c["n_chars"] // 8
    buckets.setdefault(key, []).append(c)

for k in buckets:
    random.shuffle(buckets[k])

ordered = []
for k in sorted(buckets):
    ordered.extend(buckets[k])

# Select 129 new poems
selected = ordered[:129]

# Verify no overlaps
selected_norms = set(normalize(c["text"]) for c in selected)
assert len(selected_norms & reserved_norms) == 0, "Overlap with reserved!"

# Show distribution
from collections import Counter
len_dist = Counter()
for c in selected:
    len_dist[c["n_chars"] // 8 * 8] += 1
print("Selected %d new poems:" % len(selected))
for b in sorted(len_dist):
    print("  %d-%d chars: %d" % (b, b + 7, len_dist[b]))

# Verify train_171 ⊂ train_300 structure
train171_ids = set()
with open(os.path.join(ROOT, "data", "train_171_manifest.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        train171_ids.add(r["poem_id"])

print("\ntrain_171 poems: %d" % len(train171_ids))
print("New poems: %d" % len(selected))
print("train_300 = train_171 + new = %d" % (len(train171_ids) + len(selected)))

# Save selection as a manifest that generate_paddlespeech_distillation_data.py can use
# Format: same as train_manifest.jsonl
out_path = os.path.join(ROOT, "data", "new_129_manifest.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for c in selected:
        f.write(json.dumps({
            "poem_id": c["poem_id"],
            "text": c["text"],
            "title": c["title"],
            "author": c["author"],
        }, ensure_ascii=False) + "\n")

# Save text hashes for verification
hashes_path = os.path.join(ROOT, "data", "new_129_hashes.json")
with open(hashes_path, "w", encoding="utf-8") as f:
    json.dump([c["text_hash"] for c in selected], f)

print("\nSaved: %s" % out_path)
print("Saved: %s" % hashes_path)

# Show a few examples
print("\nSample new poems:")
for c in selected[:5]:
    print("  %s (%s, %d chars): %s" % (c["poem_id"], c["author"], c["n_chars"], c["text"][:40]))
