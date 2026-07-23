"""
Step 1: Generate high-quality poetry recitation audio using MOSS-TTS 8B.

Usage:
    python 01_generate_data.py

Output:
    data/raw_audio/{poem_id}.wav

Requires:
    - MOSS-TTS 8B model (auto-downloads from HuggingFace)
    - ~8GB VRAM (FP16) or use bitsandbytes for 4-bit quantization
    - Or use llama.cpp GGUF path (see MOSS-TTS repo for details)
"""

import json
import os
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
POEMS_JSONL = Path(__file__).parent.parent.parent / "app" / "src" / "main" / "assets" / "poems_4.jsonl"

# Poems tagged with "唐诗三百首" — the training corpus
TARGET_TAG = "唐诗三百首"

# Instruction injected into 8B model for emotional poetry recitation
INSTRUCTION = "深���朗诵这首古诗，抑扬顿挫，节奏舒缓，韵律分明"

# Voice to use (from MOSS-TTS builtin voices or reference audio)
VOICE = "Junhao"

# ── Main ────────────────────────────────────────────────

def load_target_poems() -> list[dict]:
    """Load poems tagged with 唐诗三百首 from JSONL."""
    poems = []
    with open(POEMS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            poem = json.loads(line.strip())
            tags = poem.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            if TARGET_TAG in tags:
                poems.append(poem)
    print(f"Loaded {len(poems)} poems tagged '{TARGET_TAG}'")
    return poems


def generate_one(poem: dict, output_path: Path) -> bool:
    """
    Generate WAV for a single poem using MOSS-TTS 8B.
    
    Returns True on success.
    """
    text = poem["content"].replace("\n", "")
    title = poem.get("title", "unknown")
    
    print(f"  Generating: {title} ({len(text)} chars)")
    
    try:
        # Option A: Use transformers + torch (needs GPU)
        # See: https://github.com/OpenMOSS/MOSS-TTS
        #
        # from transformers import AutoModel, AutoProcessor
        # import torch, torchaudio
        #
        # model = AutoModel.from_pretrained(
        #     "OpenMOSS-Team/MOSS-TTS-v1.5",
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
        #     trust_remote_code=True,
        # )
        # processor = AutoProcessor.from_pretrained(
        #     "OpenMOSS-Team/MOSS-TTS-v1.5",
        #     trust_remote_code=True,
        # )
        # processor.audio_tokenizer = processor.audio_tokenizer.to("cuda")
        #
        # user_msg = processor.build_user_message(
        #     text=text,
        #     instruction=INSTRUCTION,
        #     quality="high",
        #     language="Chinese",
        # )
        # 
        # audio = model.generate(user_msg)
        # torchaudio.save(str(output_path), audio.unsqueeze(0), 48000)
        
        # Option B: Use llama.cpp GGUF (lower VRAM, slower)
        # See: https://github.com/OpenMOSS/llama.cpp/tree/moss-tts-firstclass
        
        print(f"    [STUB] Would generate: {text[:50]}...")
        print(f"    TODO: implement actual 8B inference")
        return False
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    poems = load_target_poems()
    if not poems:
        print("No poems found! Check POEMS_JSONL path and tag.")
        sys.exit(1)
    
    succeeded = 0
    for i, poem in enumerate(poems):
        poem_id = poem.get("sourceUrl", f"poem_{i}").split("/")[-1].replace(".aspx", "").replace("shiwenv_", "")
        output_path = RAW_AUDIO_DIR / f"{poem_id}.wav"
        
        if output_path.exists():
            print(f"  [{i+1}/{len(poems)}] Skip (exists): {output_path.name}")
            succeeded += 1
            continue
        
        print(f"  [{i+1}/{len(poems)}] ", end="")
        if generate_one(poem, output_path):
            succeeded += 1
    
    print(f"\nDone: {succeeded}/{len(poems)} poems generated")
    print(f"Output: {RAW_AUDIO_DIR}")


if __name__ == "__main__":
    main()
