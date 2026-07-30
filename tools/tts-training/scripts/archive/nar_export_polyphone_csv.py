"""Export every polyphone occurrence as CSV for manual review.

For each occurrence, includes:
- Poem title and line context
- pypinyin default reading
- All candidate readings from known polyphone table
- Teacher ASR pinyin (from clean_data_list.json)
- Selected reading (to be filled by human)
- Selection basis (to be filled)
"""
import csv
import json
import re
from pathlib import Path
from pypinyin import pinyin, Style
from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]
cc = OpenCC("t2s")

PUNCT = re.compile(r"[，。！？；：、,.!?;:\"'""''（）()\[\]【】《》〈〉…—\-．\.\n\r\s\u3000]")

# Polyphone characters to track (from scan results)
POLYPHONES = set("长重还为行少朝相更度中看当教分间从和王令将觉落处见得调思系兴衰斜")

# MFA phone format mapping (pypinyin tone3 → MFA style)
TONE_MAP = {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5"}


def normalize(text):
    return PUNCT.sub("", cc.convert(text))


def extract_content(full_text):
    parts = full_text.split("。", 1)
    return parts[1] if len(parts) > 1 else full_text


def get_pinyin_tone3(char):
    py = pinyin(char, style=Style.TONE3, heteronym=True)
    if py and py[0]:
        return py[0]
    return []


def main():
    with open(ROOT / "data" / "clean_data_list.json", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]

    # Build occurrence list
    occurrences = []

    for s_idx, s in enumerate(samples):
        content = s["content"]
        title = s["title"]
        # Split into lines by punctuation
        lines = re.split(r"[，。？！；]", content)
        lines = [l for l in lines if l.strip()]

        char_global_idx = 0
        for line_idx, line in enumerate(lines):
            for char_idx, char in enumerate(line):
                if char in POLYPHONES:
                    default_py = pinyin(char, style=Style.TONE3)
                    default = default_py[0][0] if default_py and default_py[0] else "?"
                    candidates = get_pinyin_tone3(char)

                    # Context: full line + surrounding chars
                    ctx_start = max(0, char_idx - 4)
                    ctx_end = min(len(line), char_idx + 5)
                    context = line[ctx_start:ctx_end]
                    char_in_context = line[max(0, char_idx-2):min(len(line), char_idx+3)]

                    occurrences.append({
                        "sample_index": s_idx,
                        "title": title[:25],
                        "line_index": line_idx,
                        "char_index_in_line": char_idx,
                        "char": char,
                        "context_line": line,
                        "context_around": char_in_context,
                        "pypinyin_default": default,
                        "all_candidates": "/".join(candidates),
                        "teacher_asr_hint": s.get("asr_normalized", ""),
                        "selected_pinyin": "",  # to be filled
                        "selection_basis": "unverified",  # to be filled
                        "notes": "",
                    })
                char_global_idx += 1

    # Write CSV
    csv_path = ROOT / "data" / "polyphone_review.csv"
    fieldnames = [
        "sample_index", "title", "line_index", "char_index_in_line",
        "char", "context_line", "context_around",
        "pypinyin_default", "all_candidates",
        "selected_pinyin", "selection_basis", "notes",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for occ in occurrences:
            row = {k: occ.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"Total polyphone occurrences: {len(occurrences)}")
    print(f"Unique chars: {len(set(o['char'] for o in occurrences))}")
    print(f"Output: {csv_path}")

    # Summary by char
    from collections import Counter
    char_counts = Counter(o["char"] for o in occurrences)
    print(f"\nOccurrences by character:")
    for char, count in char_counts.most_common():
        # Show default readings
        defaults = set(o["pypinyin_default"] for o in occurrences if o["char"] == char)
        print(f"  {char} ({count}x): defaults={defaults}")


if __name__ == "__main__":
    main()
