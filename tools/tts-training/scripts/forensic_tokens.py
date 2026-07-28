"""Forensic 2: Compare token distributions — PASS vs FAIL vs greedy."""
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = str(ROOT / "models" / "MOSS-TTS-Nano")
CODEC_PATH = str(ROOT / "models" / "MOSS-Audio-Tokenizer-Nano").replace("\\", "/")
CKPT = ROOT / "output" / "moss_poetry_sft_320_v4" / "checkpoint-epoch-3"

MOSS_AUDIO_TOKENIZER_TYPE = "moss-audio-tokenizer-nano"

SAMPLES = [
    ("wujue_01_jingyesi", "静夜思，唐代·李白。床前明月光，疑是地上霜。举头望明月，低头思故乡。", "PASS"),
    ("wujue_04_unseen", "江雪，唐代·柳宗元。千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。", "FAIL"),
    ("qilv_01_jinse", "锦瑟，唐代·李商隐。锦瑟无端五十弦，一弦一柱思华年。庄生晓梦迷蝴蝶，望帝春心托杜鹃。沧海月明珠有泪，蓝田日暖玉生烟。此情可待成追忆？只是当时已惘然。", "PASS"),
]


def load_model(device, dtype):
    from safetensors.torch import load_file
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)
    st = CKPT / "model.safetensors"
    if st.exists():
        state_dict = load_file(str(st))
        model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=dtype)
    if hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation("sdpa")
    model.eval()
    return model


def generate_and_analyze(model, text, device, do_sample, label):
    """Generate and capture token statistics."""
    # We need to hook into the model to capture generated tokens
    # Use inference but also capture the audio_token_ids
    result = model.inference(
        text=text,
        output_audio_path=f"/tmp/forensic_{label}.wav",
        mode="continuation",
        text_tokenizer_path=BASE_MODEL,
        audio_tokenizer_type=MOSS_AUDIO_TOKENIZER_TYPE,
        audio_tokenizer_pretrained_name_or_path=CODEC_PATH,
        device=device,
        max_new_frames=750,
        do_sample=do_sample,
        use_kv_cache=True,
    )

    audio_token_ids = result["audio_token_ids"]  # (frames, n_vq)
    if hasattr(audio_token_ids, "cpu"):
        tokens = audio_token_ids.cpu().numpy()
    else:
        tokens = audio_token_ids

    n_frames, n_vq = tokens.shape
    print(f"\n  [{label}] frames={n_frames}, n_vq={n_vq}")

    # Per-codebook stats
    for cb in range(min(n_vq, 4)):  # First 4 codebooks
        cb_tokens = tokens[:, cb]
        unique = len(set(cb_tokens.tolist()))
        counter = Counter(cb_tokens.tolist())
        top5 = counter.most_common(5)
        # Longest consecutive same token
        max_run = 1
        cur_run = 1
        for i in range(1, len(cb_tokens)):
            if cb_tokens[i] == cb_tokens[i-1]:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1

        top_str = ", ".join(f"{t}:{c}({c/n_frames*100:.0f}%)" for t, c in top5)
        print(f"    CB{cb}: unique={unique}, max_run={max_run}, top5=[{top_str}]")

    return tokens


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    model = load_model(device, dtype)

    for pid, text, expected in SAMPLES:
        print(f"\n{'='*60}")
        print(f"  {pid} ({expected})")
        print(f"{'='*60}")

        # Sampling
        print(f"\n  --- do_sample=True ---")
        try:
            gen_sample = generate_and_analyze(model, text, device, do_sample=True, label=f"{pid}_sample")
        except Exception as e:
            print(f"  FAILED: {e}")
            gen_sample = None

        # Greedy
        print(f"\n  --- do_sample=False (greedy) ---")
        try:
            gen_greedy = generate_and_analyze(model, text, device, do_sample=False, label=f"{pid}_greedy")
        except Exception as e:
            print(f"  FAILED: {e}")
            gen_greedy = None

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
