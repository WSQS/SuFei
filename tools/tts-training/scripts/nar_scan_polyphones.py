"""Scan all 221 clean poems for polyphone characters that need manual review."""
import json
import re
from pathlib import Path
from collections import Counter
from pypinyin import pinyin, Style
from pypinyin.pinyin_dict import pinyin_dict

ROOT = Path(__file__).resolve().parents[1]

# Known polyphone characters common in classical Chinese poetry
# that pypinyin may not disambiguate correctly
KNOWN_POLYPHONES = {
    "长": {"cháng": "long", "zhǎng": "grow"},
    "重": {"chóng": "again", "zhòng": "heavy"},
    "还": {"huán": "return", "hái": "still"},
    "为": {"wèi": "for", "wéi": "be"},
    "行": {"xíng": "walk", "háng": "row"},
    "少": {"shǎo": "few", "shào": "young"},
    "朝": {"zhāo": "morning", "cháo": "dynasty"},
    "相": {"xiāng": "mutual", "xiàng": "appearance"},
    "更": {"gēng": "change", "gèng": "more"},
    "度": {"dù": "degree", "duó": "guess"},
    "中": {"zhōng": "middle", "zhòng": "hit"},
    "看": {"kàn": "look", "kān": "guard"},
    "当": {"dāng": "should", "dàng": "proper"},
    "教": {"jiāo": "teach", "jiào": "religion"},
    "分": {"fēn": "divide", "fèn": "part"},
    "间": {"jiān": "between", "jiàn": "gap"},
    "从": {"cóng": "follow", "zòng": "loose"},
    "和": {"hé": "and", "hè": "echo"},
    "王": {"wáng": "king", "wàng": "reign"},
    "令": {"lìng": "order", "líng": "name"},
    "将": {"jiāng": "will", "jiàng": "general"},
    "觉": {"jué": "feel", "jiào": "sleep"},
    "落": {"luò": "fall", "là": "miss"},
    "处": {"chù": "place", "chǔ": "dwell"},
    "见": {"jiàn": "see", "xiàn": "appear"},
    "得": {"dé": "get", "děi": "must"},
    "调": {"diào": "tune", "tiáo": "adjust"},
    "思": {"sī": "think", "sāi": "rustic"},
    "系": {"xì": "tie", "jì": "connect"},
    "兴": {"xìng": "interest", "xīng": "rise"},
    "衰": {"cuī": "decline (poetic)", "shuāi": "decline (modern)"},
    "斜": {"xié": "slant (modern)", "xiá": "slant (poetic)"},
}


def main():
    with open(ROOT / "data" / "clean_data_list.json", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]

    # Collect all unique characters
    all_chars = Counter()
    for s in samples:
        for ch in s["content"]:
            if "\u4e00" <= ch <= "\u9fff":
                all_chars[ch] += 1

    print(f"Total unique Chinese chars: {len(all_chars)}")

    # Find polyphone characters
    found_polyphones = []
    for ch, count in all_chars.items():
        if ch in KNOWN_POLYPHONES:
            # Get pypinyin's default reading
            py = pinyin(ch, style=Style.TONE3, heteronym=True)
            readings = set()
            if py and py[0]:
                readings = set(py[0])
            found_polyphones.append({
                "char": ch,
                "count": count,
                "known_readings": KNOWN_POLYPHONES[ch],
                "pypinyin_readings": readings,
            })

    print(f"\nKnown polyphone characters found: {len(found_polyphones)}")
    print(f"{'Char':<6} {'Count':>5} {'Known readings':<40} {'pypinyin default':<20} Poems")
    print("-" * 120)

    for p in sorted(found_polyphones, key=lambda x: -x["count"]):
        # Find poems containing this char
        poems = []
        for s in samples:
            if p["char"] in s["content"]:
                title = s["title"].split("/")[0].strip()[:10]
                poems.append(title)
        poems_str = ", ".join(poems[:5])
        if len(poems) > 5:
            poems_str += f" (+{len(poems)-5})"

        readings_str = " / ".join(f"{k}({v})" for k, v in p["known_readings"].items())
        py_str = ", ".join(p["pypinyin_readings"])
        print(f"{p['char']:<6} {p['count']:>5} {readings_str:<40} {py_str:<20} {poems_str}")

    # Build override table
    print(f"\n{'='*60}")
    print("Polyphone override table (to be manually reviewed)")
    print(f"{'='*60}")
    print("{")
    for p in sorted(found_polyphones, key=lambda x: x["char"]):
        # For each poem, show context
        contexts = []
        for s in samples:
            content = s["content"]
            idx = content.find(p["char"])
            if idx >= 0:
                start = max(0, idx - 3)
                end = min(len(content), idx + 4)
                ctx = content[start:end]
                contexts.append(f'"{ctx}"')
        contexts_str = ", ".join(contexts[:3])
        print(f'  "{p["char"]}": {{  # {p["known_readings"]}')
        print(f'    # contexts: {contexts_str}')
        print(f'  }},')
    print("}")


if __name__ == "__main__":
    main()
