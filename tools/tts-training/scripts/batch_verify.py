r"""
Batch verify fine-tuned MOSS-TTS-Nano vs base model.

Generates WAVs for a set of representative poems with multiple checkpoints,
plus reports training data statistics.

Usage (from tools/tts-training):
    python scripts\batch_verify.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "verify_v4"
BASE_MODEL = ROOT / "models" / "MOSS-TTS-Nano"
CODEC_PATH = ROOT / "models" / "MOSS-Audio-Tokenizer-Nano"

# Checkpoints to compare: (label, path)
CHECKPOINTS = [
    ("base",                BASE_MODEL),
    ("320v3ep3",            ROOT / "output" / "moss_poetry_sft_320_v3" / "checkpoint-epoch-3"),
    ("320v4ep3",            ROOT / "output" / "moss_poetry_sft_320_v4" / "checkpoint-epoch-3"),
]

# Representative test poems — text must match teacher format: title，dynasty·author。content
TEST_POEMS = [
    ("short_jingyesi", "静夜思，唐代·李白。床前明月光，疑是地上霜。举头望明月，低头思故乡。"),
    ("medium_wangyue", "望岳，唐代·杜甫。岱宗夫如何？齐鲁青未了。造化钟神秀，阴阳割昏晓。荡胸生曾云，决眦入归鸟。会当凌绝顶，一览众山小。"),
    ("medium_jinse", "锦瑟，唐代·李商隐。锦瑟无端五十弦，一弦一柱思华年。庄生晓梦迷蝴蝶，望帝春心托杜鹃。沧海月明珠有泪，蓝田日���玉生烟。此情可待成追忆？只是当时已惘然。"),
]

MOSS_AUDIO_TOKENIZER_TYPE = "moss-audio-tokenizer-nano"


def print_training_stats():
    print("=" * 60)
    print("Training Data Statistics")
    print("=" * 60)

    jsonl_path = DATA_DIR / "train_with_codes.jsonl"
    if not jsonl_path.exists():
        print(f"  [WARN] {jsonl_path} not found")
        return

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    text_lengths = [len(r["text"]) for r in records]
    code_lengths = [len(r.get("audio_codes", [])) for r in records]

    import statistics
    print(f"  Total samples:     {len(records)}")
    print(f"  Text length:       min={min(text_lengths)}, max={max(text_lengths)}, "
          f"mean={statistics.mean(text_lengths):.1f}, median={statistics.median(text_lengths):.0f}")
    print(f"  Audio code length: min={min(code_lengths)}, max={max(code_lengths)}, "
          f"mean={statistics.mean(code_lengths):.1f}, median={statistics.median(code_lengths):.0f}")

    print(f"\n  Text length distribution:")
    buckets = [(0, 30, "short"), (30, 60, "medium"), (60, 120, "long"), (120, 500, "very_long")]
    for lo, hi, label in buckets:
        count = sum(1 for l in text_lengths if lo <= l < hi)
        bar = "#" * (count // 5)
        print(f"    {label:>10} ({lo}-{hi}): {count:4d} {bar}")

    print()


def load_model(checkpoint_path: str, device: torch.device, dtype: torch.dtype, base_model_path: str = None):
    checkpoint_path = str(checkpoint_path)
    if base_model_path and checkpoint_path != base_model_path:
        from safetensors.torch import load_file
        model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
        st_path = Path(checkpoint_path) / "model.safetensors"
        if st_path.exists():
            print(f"  Loading weights from {st_path.name} ...")
            state_dict = load_file(str(st_path))
            model.load_state_dict(state_dict, strict=True)
        else:
            raise FileNotFoundError(f"No model.safetensors in {checkpoint_path}")
    else:
        print(f"  Loading {checkpoint_path} ...")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
    model.to(device=device, dtype=dtype)
    if hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation("sdpa" if device.type == "cuda" else "eager")
    model.eval()
    return model


def generate_sample(model, text: str, output_path: str, model_path: str, codec_path: str, device, tag: str):
    print(f"  [{tag}] {text[:25]}... -> {Path(output_path).name}")
    t0 = time.time()
    result = model.inference(
        text=text,
        output_audio_path=output_path,
        mode="continuation",
        text_tokenizer_path=model_path,
        audio_tokenizer_type=MOSS_AUDIO_TOKENIZER_TYPE,
        audio_tokenizer_pretrained_name_or_path=codec_path,
        device=device,
        max_new_frames=1500,
        do_sample=True,
        audio_repetition_penalty=1.2,
        use_kv_cache=True,
    )
    elapsed = time.time() - t0
    frames = int(result["audio_token_ids"].shape[0])
    print(f"    {elapsed:.1f}s, {frames} frames")
    return result


def main():
    print_training_stats()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    codec_path_str = str(CODEC_PATH).replace("\\", "/")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter to existing checkpoints
    active = [(label, path) for label, path in CHECKPOINTS if path.exists()]
    print(f"Comparing {len(active)} checkpoints: {[l for l, _ in active]}")
    print()

    # Load one model at a time, generate all test poems, then unload
    base_model_str = str(BASE_MODEL)
    for ckpt_label, ckpt_path in active:
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_label}")
        print(f"{'='*60}")

        is_base = (ckpt_label == "base")
        model = load_model(
            str(ckpt_path), device, dtype,
            base_model_path=base_model_str if not is_base else None,
        )
        model_path_str = str(ckpt_path).replace("\\", "/")

        for name, text in TEST_POEMS:
            out_file = OUTPUT_DIR / f"{name}_{ckpt_label}.wav"
            try:
                generate_sample(model, text, str(out_file), base_model_str, codec_path_str, device, ckpt_label)
            except Exception as e:
                print(f"    FAILED: {e}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Poem':<25}", end="")
    for label, _ in active:
        print(f"{label:>15}", end="")
    print()
    print("-" * (25 + 15 * len(active)))

    for name, text in TEST_POEMS:
        print(f"{name:<25}", end="")
        for label, _ in active:
            wav = OUTPUT_DIR / f"{name}_{label}.wav"
            if wav.exists():
                size_kb = wav.stat().st_size / 1024
                print(f"{size_kb:>12.0f}KB", end="   ")
            else:
                print(f"{'--':>15}", end="")
        print()

    print(f"\nAll outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
