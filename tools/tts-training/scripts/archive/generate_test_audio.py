"""Generate test audio for all 24 poems with MOSS v4 checkpoint."""
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
TEST_SET = ROOT / "data" / "test_set_v1.json"
BASE_MODEL = ROOT / "models" / "MOSS-TTS-Nano"
CODEC_PATH = ROOT / "models" / "MOSS-Audio-Tokenizer-Nano"
CKPT = ROOT / "output" / "moss_poetry_sft_320_v4" / "checkpoint-epoch-3"
OUTPUT_DIR = ROOT / "output" / "test_audio_v4_greedy"

MOSS_AUDIO_TOKENIZER_TYPE = "moss-audio-tokenizer-nano"


def load_model(device, dtype):
    from safetensors.torch import load_file
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL), trust_remote_code=True)
    st_path = CKPT / "model.safetensors"
    if st_path.exists():
        print(f"Loading v4 weights from {st_path.parent.name}...")
        state_dict = load_file(str(st_path))
        model.load_state_dict(state_dict, strict=True)
    else:
        print("WARNING: No safetensors checkpoint found, using base model")
    model.to(device=device, dtype=dtype)
    if hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation("sdpa")
    model.eval()
    return model


def main():
    with open(TEST_SET, encoding="utf-8") as f:
        test_set = json.load(f)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    codec_str = str(CODEC_PATH).replace("\\", "/")
    base_str = str(BASE_MODEL).replace("\\", "/")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(device, dtype)

    for s in test_set["samples"]:
        pid = s["poem_id"]
        title = s["title"]
        author = s["author"]
        dynasty = s["dynasty"]
        content = s["content"]

        # Build full text matching teacher format
        full_text = f"{title}，{dynasty}·{author}。{content}"

        out_file = OUTPUT_DIR / f"{pid}_v4_greedy.wav"

        print(f"[{pid}] {title}...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = model.inference(
                text=full_text,
                output_audio_path=str(out_file),
                mode="continuation",
                text_tokenizer_path=base_str,
                audio_tokenizer_type=MOSS_AUDIO_TOKENIZER_TYPE,
                audio_tokenizer_pretrained_name_or_path=codec_str,
                device=device,
                max_new_frames=750,
                do_sample=False,
                use_kv_cache=True,
            )
            elapsed = time.time() - t0
            frames = int(result["audio_token_ids"].shape[0])
            print(f"{elapsed:.1f}s, {frames} frames")
        except Exception as e:
            print(f"FAILED: {e}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nAll outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
