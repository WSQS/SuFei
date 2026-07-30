"""Select batch 2: 80 more poems to cover 53 failures."""
import json, os, random, re, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def normalize(text):
    return re.sub(r"[，。？！；：、·\s\n]", "", text)

def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# All reserved
reserved_norms = set()
for mf in ["train_171_manifest.jsonl", "holdout_20_manifest.jsonl",
           "external_unseen_6_manifest.jsonl", "train_manifest.jsonl"]:
    path = os.path.join(ROOT, "data", mf)
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            reserved_norms.add(normalize(r["text"]))

# Exclude batch 1 (all 129, pass or fail)
batch1 = [json.loads(l) for l in open(
    os.path.join(ROOT, "data", "new_129_manifest.jsonl"), encoding="utf-8")]
for c in batch1:
    reserved_norms.add(normalize(c["text"]))

print("Reserved: %d" % len(reserved_norms))

# Load candidates from assets
assets = os.path.join(
    os.path.dirname(os.path.dirname(ROOT)),
    "app", "src", "main", "assets",
)

candidates = []
seen = set()
for i in range(5):
    path = os.path.join(assets, "poems_%d.jsonl" % i)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("dynasty") != "唐代":
                continue
            content = r.get("content", "")
            chars = [c for c in content if "\u4e00" <= c <= "\u9fff"]
            n = len(chars)
            if not (20 <= n <= 56):
                continue
            title = r["title"].strip()
            author = r["author"].strip()
            full = "%s，%s·%s。%s" % (title, "唐代", author, content.replace("\n", ""))
            norm = normalize(full)
            if norm in reserved_norms or norm in seen:
                continue
            seen.add(norm)
            candidates.append({"text": full, "n_chars": n})

print("Batch 2 candidates: %d" % len(candidates))

random.seed(137)
random.shuffle(candidates)

# Select 80 (need 53 to pass; assume ~60% pass rate → 80 gives ~48;
# be generous: select 90)
selected = candidates[:90]

from collections import Counter
ld = Counter(c["n_chars"] // 8 * 8 for c in selected)
print("Selected %d:" % len(selected))
for b in sorted(ld):
    print("  %d-%d: %d" % (b, b + 7, ld[b]))

out = os.path.join(ROOT, "data", "new_batch2_manifest.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for i, c in enumerate(selected):
        pid = "poem_b2_%04d" % i
        f.write(json.dumps({"poem_id": pid, "text": c["text"]}, ensure_ascii=False) + "\n")
print("Saved: %s" % out)
