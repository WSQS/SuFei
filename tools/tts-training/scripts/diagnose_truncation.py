"""Diagnose training data: compute packed_length and truncation stats."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MOSS-TTS-Nano"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from finetuning.dataset import MossTTSNanoSFTDataset
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data" / "train_with_codes.jsonl"
MODEL_PATH = ROOT / "models" / "MOSS-TTS-Nano"
MAX_LENGTH = 2048


def main():
    # Load model config only (no weights needed)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    cfg = model.config
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

    records = []
    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    print(f"Records: {len(records)}")
    print(f"max_length: {MAX_LENGTH}")
    print(f"n_vq: {cfg.n_vq}")
    print()

    stats = []
    truncated = 0
    lost_eos = 0

    for i, record in enumerate(records):
        ds = MossTTSNanoSFTDataset(records=[dict(record)], tokenizer=tokenizer, model_config=cfg, max_length=MAX_LENGTH)
        try:
            example = ds[0]
        except Exception as e:
            print(f"  [ERROR] record {i}: {e}")
            continue

        seq_len = int(example["seq_len"])
        prompt_len = int(example["prompt_length"])
        target_len = seq_len - prompt_len

        # Reconstruct full untruncated length
        target_codes = record.get("audio_codes", [])
        target_frames = len(target_codes)
        # Each frame is n_vq tokens
        target_audio_tokens = target_frames * cfg.n_vq
        packed_full = prompt_len + target_audio_tokens + 1  # +1 for audio_end_token
        is_truncated = packed_full > MAX_LENGTH
        has_eos = not is_truncated or (seq_len == MAX_LENGTH and False)  # simplified

        # Check if EOS is in the sequence
        full_ids = example["full_input_ids"].tolist()
        last_token = full_ids[-1]
        eos_id = cfg.audio_end_token_id
        has_eos = (last_token == eos_id)

        text_preview = record["text"][:30]
        stats.append({
            "idx": i,
            "text_len": len(record["text"]),
            "prompt_len": prompt_len,
            "target_frames": target_frames,
            "packed_full": packed_full,
            "seq_len": seq_len,
            "truncated": is_truncated,
            "has_eos": has_eos,
            "title": text_preview,
        })

        if is_truncated:
            truncated += 1
        if not has_eos:
            lost_eos += 1

    print(f"{'Idx':>4} {'Text':>5} {'Prompt':>7} {'Frames':>7} {'Packed':>7} {'SeqLen':>7} {'Trunc':>6} {'EOS':>5} Title")
    print("-" * 90)
    for s in stats:
        flag_t = "***" if s["truncated"] else ""
        flag_e = "!!!" if not s["has_eos"] else ""
        print(f"{s['idx']:4d} {s['text_len']:5d} {s['prompt_len']:7d} {s['target_frames']:7d} "
              f"{s['packed_full']:7d} {s['seq_len']:7d} {flag_t:>6} {flag_e:>5} {s['title']}")

    print()
    total = len(stats)
    print(f"Summary:")
    print(f"  Total:          {total}")
    print(f"  Truncated:      {truncated} ({truncated/total*100:.1f}%)")
    print(f"  Missing EOS:    {lost_eos} ({lost_eos/total*100:.1f}%)")

    import statistics
    packed = [s["packed_full"] for s in stats]
    prompts = [s["prompt_len"] for s in stats]
    print(f"  Packed length:  min={min(packed)}, max={max(packed)}, "
          f"mean={statistics.mean(packed):.0f}, median={statistics.median(packed):.0f}")
    print(f"  Prompt length:  min={min(prompts)}, max={max(prompts)}, "
          f"mean={statistics.mean(prompts):.0f}, median={statistics.median(prompts):.0f}")


if __name__ == "__main__":
    main()
