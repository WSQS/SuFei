"""Build MFA corpus with character-level tokens and custom dictionary.

Strategy:
- Each Chinese character becomes a unique token: char_pinyin (e.g. 春_chun1)
- Pinyin comes from audited polyphone review CSV (overrides pypinyin)
- .lab files use space-separated tokens
- Custom dictionary maps each token to MFA phone sequence
- Punctuation becomes silence tokens
"""
import csv
import json
import re
from pathlib import Path

from pypinyin import pinyin, Style
from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]
CLEAN_JSON = ROOT / "data" / "clean_data_list.json"
POLYPHONE_CSV = ROOT / "data" / "polyphone_review_audited.csv"
MFA_DICT = ROOT / "data" / "mfa_char_dictionary.txt"
CORPUS_DIR = ROOT / "data" / "mfa_corpus_char"

cc = OpenCC("t2s")

# Punctuation → silence tokens
PUNCT_TO_TOKEN = {
    "，": "<comma>",
    "。": "<period>",
    "？": "<question>",
    "！": "<exclamation>",
    "；": "<semicolon>",
    "：": "<colon>",
}


def load_polyphone_overrides():
    """Load audited polyphone CSV into lookup: (sample_index, char) → pinyin."""
    overrides = {}
    if not POLYPHONE_CSV.exists():
        print("WARNING: polyphone_review_audited.csv not found")
        return overrides

    with open(POLYPHONE_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("review_status", "").strip()
            if status == "需要修改" or status == "需修改且有变体":
                sidx = int(row["sample_index"])
                ch = row["char"]
                selected = row.get("selected_pinyin", "").strip()
                if selected:
                    overrides[(sidx, ch)] = selected
    return overrides


def char_to_pinyin(char, sample_index=None, overrides=None):
    """Get pinyin for a character, using override if available."""
    # Check override first
    if overrides and (sample_index, char) in overrides:
        return overrides[(sample_index, char)]

    # Default pypinyin
    py = pinyin(char, style=Style.TONE3, errors="default")
    if py and py[0] and py[0][0] != char:
        return py[0][0]

    # Non-Chinese char
    return None


def pinyin_to_mfa_phones(py):
    """Convert pinyin (tone3 format like 'chun1') to MFA phone sequence.

    MFA mandarin_china_mfa uses phones like:
    a1 a2 a3 a4 a5 (finals with tone)
    b p m f d t n l g k h j q x zh ch sh r z c s (initials)

    We need to split pinyin into initial + final + tone,
    matching the mandarin_china_mfa phone set.
    """
    if not py or len(py) < 2:
        return ["sil"]

    # Extract tone
    tone = ""
    base = py
    if py[-1].isdigit():
        tone = py[-1]
        base = py[:-1]
    else:
        tone = "5"  # neutral tone

    # Handle y/w pseudo-initials
    if base.startswith("y"):
        if base == "yu":
            base = "v"
        elif base == "yue":
            base = "ve"
        elif base == "yuan":
            base = "van"
        elif base == "yun":
            base = "vn"
        elif base == "you":
            base = "iou"
        elif base == "yong":
            base = "iong"
        elif len(base) > 1 and base[1] in "ae":
            base = "i" + base[1:]
        elif len(base) > 1 and base[1] == "i":
            base = base[1:]
        else:
            base = "i" + base[1:] if len(base) > 1 else "i"
    elif base.startswith("w"):
        if len(base) == 1:
            base = "u"
        else:
            base = "u" + base[1:]

    # Fix pypinyin-specific finals
    final_fixes = {
        "un": "uen", "ui": "uei", "iu": "iou", "ue": "ve",
    }
    base = final_fixes.get(base, base)

    # Split into initial + final
    initials = {"b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
                "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s"}

    init = ""
    final_base = base

    for ilen in [2, 1]:
        if len(base) > ilen:
            candidate = base[:ilen]
            if candidate in initials:
                init = candidate
                final_base = base[ilen:]
                break

    # Disambiguate i → i/ii/iii
    if final_base.rstrip("12345") == "i" or final_base.rstrip("12345") == "":
        if init in {"j", "q", "x"} and final_base.rstrip("12345") == "":
            final_base = "i" + tone
        elif init in {"z", "c", "s"} and final_base.rstrip("12345") == "":
            final_base = "ii" + tone
        elif init in {"zh", "ch", "sh", "r"} and final_base.rstrip("12345") == "":
            final_base = "iii" + tone

    # Build phone sequence
    phones = []
    if init:
        phones.append(init)
    if final_base:
        phones.append(final_base + tone)

    if not phones:
        phones = ["sil"]

    return phones


def build_dictionary(entries):
    """Build MFA dictionary file from char_pinyin → phone mappings."""
    lines = []
    for token, phones in sorted(entries.items()):
        lines.append(f"{token}\t{' '.join(phones)}")
    return "\n".join(lines)


def main():
    overrides = load_polyphone_overrides()
    print(f"Loaded {len(overrides)} polyphone overrides")

    with open(CLEAN_JSON, encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    dictionary_entries = {}  # token → phone list

    for s in samples:
        sidx = s["index"]
        full_text = s["full_text"]
        title = s["title"]

        # Build token sequence
        tokens = []
        for char in full_text:
            if char in PUNCT_TO_TOKEN:
                tokens.append(PUNCT_TO_TOKEN[char])
            elif "\u4e00" <= char <= "\u9fff":
                py = char_to_pinyin(char, sidx, overrides)
                if py:
                    token = f"{char}_{py}"
                    tokens.append(token)
                    # Add to dictionary if not seen
                    if token not in dictionary_entries:
                        phones = pinyin_to_mfa_phones(py)
                        dictionary_entries[token] = phones
                else:
                    print(f"  WARN: no pinyin for {char} in {title}")
            elif char in "·／":
                tokens.append("<sp>")  # short pause
            else:
                # Skip other chars (spaces, etc)
                pass

        # Write .lab file
        poem_id = f"poem_{sidx:04d}"
        lab_content = " ".join(tokens)
        lab_path = CORPUS_DIR / f"{poem_id}.lab"
        lab_path.write_text(lab_content, encoding="utf-8")

    # Add silence tokens to dictionary
    dictionary_entries["<comma>"] = ["sil"]
    dictionary_entries["<period>"] = ["sil"]
    dictionary_entries["<question>"] = ["sil"]
    dictionary_entries["<exclamation>"] = ["sil"]
    dictionary_entries["<semicolon>"] = ["sil"]
    dictionary_entries["<colons>"] = ["sil"]
    dictionary_entries["<sp>"] = ["sp"]

    # Write dictionary
    dict_content = build_dictionary(dictionary_entries)
    MFA_DICT.write_text(dict_content, encoding="utf-8")
    print(f"\nDictionary: {len(dictionary_entries)} entries → {MFA_DICT}")
    print(f"Corpus: {len(samples)} .lab files in {CORPUS_DIR}")

    # Copy WAV files (reuse from existing mfa_corpus)
    import shutil
    src_wav_dir = ROOT / "data" / "mfa_corpus"
    for s in samples:
        poem_id = f"poem_{s['index']:04d}"
        src = src_wav_dir / f"{poem_id}.wav"
        dst = CORPUS_DIR / f"{poem_id}.wav"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    wavs = list(CORPUS_DIR.glob("*.wav"))
    labs = list(CORPUS_DIR.glob("*.lab"))
    print(f"WAVs: {len(wavs)}, LABs: {len(labs)}")


if __name__ == "__main__":
    main()
