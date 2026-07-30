#!/usr/bin/env python
"""
Generate high-quality poetry recitation audio using CosyVoice 3.

Usage:
    cd ~/CosyVoice
    python generate_poetry.py [--max N] [--min-len N] [--instruction natural|emotional|none]

Output:
    data/raw_audio/{poem_id}.wav
"""

import sys
import os
import json
import re
import time
from pathlib import Path

COSYVOICE_DIR = Path.home() / "CosyVoice"
sys.path.insert(0, str(COSYVOICE_DIR))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

OUTPUT_DIR = COSYVOICE_DIR / "data" / "raw_audio"
POEMS_DIR = Path("/mnt/c/Users/wsqsy/Documents/android/SuFei/app/src/main/assets")
TARGET_TAG = "唐诗三百首"
MODEL_DIR = str(COSYVOICE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B")
SAMPLE_RATE = 24000

INSTRUCTION_NATURAL = "使用自然、克制的中文诗歌朗读方式。声音沉静而清晰。"
INSTRUCTION_EMOTIONAL = "深情朗诵，语速稍慢，带有思念之情。"

PROMPT_WAV = str(COSYVOICE_DIR / "asset" / "zero_shot_prompt.wav")
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"


def load_poems(max_poems=None):
    poems = []
    for jsonl in sorted(POEMS_DIR.glob("poems_*.jsonl")):
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                poem = json.loads(line.strip())
                tags = poem.get("tags", [])
                if isinstance(tags, str):
                    tags = [tags]
                if TARGET_TAG in tags:
                    poems.append(poem)
    seen = set()
    unique = []
    for p in poems:
        url = p.get("sourceUrl", "")
        if url not in seen:
            seen.add(url)
            unique.append(p)
    if max_poems:
        unique = unique[:max_poems]
    print(f"Loaded {len(unique)} poems")
    return unique


def poem_id(poem, index):
    url = poem.get("sourceUrl", f"poem_{index}")
    return url.split("/")[-1].replace(".aspx", "").replace("shiwenv_", "")


def clean_text(text):
    text = text.replace("\n", "")
    text = re.sub(r"[（(][^)）]*[)）]", "", text)
    return text.strip()


def build_text(poem):
    """Build full text with title/author prefix for context."""
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

    return f"{prefix}{content}", content, title


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=320)
    parser.add_argument("--min-len", type=int, default=20,
                       help="Minimum content length (title prefix not counted)")
    parser.add_argument("--instruction", type=str, default="natural",
                       choices=["natural", "emotional", "none"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    instruction = {
        "natural": INSTRUCTION_NATURAL,
        "emotional": INSTRUCTION_EMOTIONAL,
        "none": None,
    }[args.instruction]

    print("Loading CosyVoice3 model...")
    t0 = time.time()
    model = AutoModel(model_dir=MODEL_DIR)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    poems = load_poems(max_poems=args.max)
    succeeded = 0
    skipped = 0

    for i, poem in enumerate(poems):
        pid = poem_id(poem, i)
        out_path = OUTPUT_DIR / f"{pid}.wav"
        if out_path.exists():
            succeeded += 1
            continue

        _, content, title = build_text(poem)
        if len(content) < args.min_len:
            skipped += 1
            continue

        full_text, _, _ = build_text(poem)
        print(f"  [{i+1}/{len(poems)}] {title} ({len(content)} chars)")

        try:
            if instruction:
                chunks = list(model.inference_instruct2(
                    full_text,
                    instruct_text=f"You are a helpful assistant. {instruction}<|endofprompt|>",
                    prompt_wav=PROMPT_WAV,
                ))
            else:
                chunks = list(model.inference_zero_shot(
                    full_text,
                    prompt_text=PROMPT_TEXT,
                    prompt_wav=PROMPT_WAV,
                ))

            if not chunks:
                print(f"    WARNING: No output")
                continue

            audio = chunks[0]["tts_speech"]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            torchaudio.save(str(out_path), audio.cpu(), SAMPLE_RATE)
            duration = audio.shape[-1] / SAMPLE_RATE
            print(f"    Saved {duration:.1f}s")
            succeeded += 1

        except Exception as e:
            print(f"    ERROR: {e}")

        torch.cuda.empty_cache()

    print(f"\nDone: {succeeded} succeeded, {skipped} skipped (too short)")


if __name__ == "__main__":
    main()
