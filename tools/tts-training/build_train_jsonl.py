#!/usr/bin/env python
"""
Build training JSONL for MOSS-TTS-Nano finetuning.

Reads poem data from assets, matches with generated WAV files,
and outputs train_raw.jsonl in MOSS format.

Output format:
    {"audio": "./audio/xxx.wav", "text": "...", "language": "zh", "instruction": "..."}
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
AUDIO_DIR = DATA_DIR / "audio"
POEMS_DIR = Path(__file__).parent.parent.parent / "app" / "src" / "main" / "assets"
TARGET_TAG = "唐诗三百首"

# Map sourceUrl hash to wav filename
# 古诗十九首 are named like "01_hash.wav", 唐诗 like "hash.wav"


def clean_text(text):
    text = text.replace("\n", "")
    text = re.sub(r"[（(][^)）]*[)）]", "", text)
    return text.strip()


def build_full_text(poem):
    """Build text matching what the teacher model (CosyVoice 3) was given.

    Format: {title}，{dynasty}·{author}。{content}
    Must stay in sync with generate_poetry.py build_text().
    """
    content = clean_text(poem["content"])
    title = poem.get("title", "")
    author = poem.get("author", "")
    dynasty = poem.get("dynasty", "")

    prefix_parts = [title]
    if dynasty and author:
        prefix_parts.append(f"{dynasty}·{author}")
    elif author:
        prefix_parts.append(author)
    prefix = "，".join(prefix_parts) + "。"

    return f"{prefix}{content}"


def load_all_poems():
    """Load all poems that have generated WAV files."""
    poems = {}
    for jsonl in sorted(POEMS_DIR.glob("poems_*.jsonl")):
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line.strip())
                tags = p.get("tags", [])
                if isinstance(tags, str):
                    tags = [tags]
                url = p.get("sourceUrl", "")
                # Extract hash from URL
                hid = url.split("/")[-1].replace(".aspx", "").replace("shiwenv_", "")
                if hid and hid not in poems:
                    poems[hid] = p
    return poems


def find_wav_for_poem(hid, audio_files):
    """Find WAV file that matches this poem's hash."""
    for wav in audio_files:
        if hid in wav.name:
            return wav
    return None


def main():
    audio_files = list(AUDIO_DIR.glob("*.wav"))
    print(f"Found {len(audio_files)} WAV files")

    poems = load_all_poems()
    print(f"Loaded {len(poems)} unique poems")

    # Build training samples
    samples = []
    for wav in sorted(audio_files):
        # Extract hash from filename (strip prefix like "01_")
        name = wav.stem
        # Try to find matching poem
        matched = False
        for hid, poem in poems.items():
            if hid in name:
                text = build_full_text(poem)
                if len(text) < 10:
                    continue
                sample = {
                    "audio": f"./audio/{wav.name}",
                    "text": text,
                    "language": "zh",
                }
                samples.append(sample)
                print(f"  {wav.name} -> {poem['title']} ({len(text)} chars)")
                matched = True
                break
        if not matched:
            print(f"  {wav.name} -> NO MATCH (skipping)")

    # Write JSONL
    output_path = DATA_DIR / "train_raw.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
