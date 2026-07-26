import json
from pathlib import Path
from collections import Counter

records = []
with open(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training\data\train_with_codes.jsonl", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line.strip()))

for r in records:
    text = r["text"]
    parts = text.split("。", 1)
    content = parts[1] if len(parts) > 1 else text
    r["_content"] = content
    r["_content_len"] = len(content)

    cl = r["_content_len"]
    if cl <= 24:
        r["_form"] = "五绝"
    elif cl <= 36:
        r["_form"] = "七绝"
    elif cl <= 60:
        r["_form"] = "五律"
    elif cl <= 80:
        r["_form"] = "七律"
    else:
        r["_form"] = "长诗"

forms = Counter(r["_form"] for r in records)
print("Form distribution:")
for f, c in sorted(forms.items()):
    print(f"  {f}: {c}")

print("\n=== Candidates for test set ===")
for form in ["五绝", "七绝", "五律", "七律", "长诗"]:
    print(f"\n--- {form} ---")
    candidates = [r for r in records if r["_form"] == form]
    for r in candidates[:8]:
        title = r["text"].split("，")[0]
        cl = r["_content_len"]
        preview = r["_content"][:30]
        print(f"  {title:20s} | len={cl:4d} | {preview}")
