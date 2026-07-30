"""Build MFA corpus with pure Chinese character tokens (no pinyin suffix).

Each character separated by space. Punctuation → Chinese punctuation tokens
that mandarin_china_mfa dictionary already contains.
Uses --no_tokenization so MFA won't re-segment.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN_JSON = ROOT / "data" / "clean_data_list.json"
CORPUS_DIR = ROOT / "data" / "mfa_corpus_plain"

# Punctuation to keep (mandarin_china_mfa has these as entries)
KEEP_PUNCT = set("，。？！；：·")


def main():
    with open(CLEAN_JSON, encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    import shutil
    src_wav_dir = ROOT / "data" / "mfa_corpus"

    for s in samples:
        sidx = s["index"]
        full_text = s["full_text"]

        # Build space-separated character sequence
        tokens = []
        for char in full_text:
            if "\u4e00" <= char <= "\u9fff":
                tokens.append(char)
            elif char in KEEP_PUNCT:
                tokens.append(char)
            elif char in "／/":
                tokens.append("·")  # normalize separator
            # skip other chars (spaces, etc)

        poem_id = f"poem_{sidx:04d}"
        lab_content = " ".join(tokens)
        (CORPUS_DIR / f"{poem_id}.lab").write_text(lab_content, encoding="utf-8")

        # Copy WAV
        src = src_wav_dir / f"{poem_id}.wav"
        dst = CORPUS_DIR / f"{poem_id}.wav"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    labs = list(CORPUS_DIR.glob("*.lab"))
    wavs = list(CORPUS_DIR.glob("*.wav"))
    print(f"Corpus: {len(labs)} LABs, {len(wavs)} WAVs in {CORPUS_DIR}")

    # Show samples
    for f in sorted(CORPUS_DIR.glob("*.lab"))[:3]:
        print(f"\n{f.name}:")
        content = f.read_text(encoding="utf-8")
        print(f"  {content[:100]}")


if __name__ == "__main__":
    main()
