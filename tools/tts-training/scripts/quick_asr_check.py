"""Quick ASR check on a single generated audio."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_tts import load_asr_model, transcribe, normalize_text, extract_content, align


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to WAV file")
    parser.add_argument("--ref", default="春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。")
    args = parser.parse_args()

    ref_text = normalize_text(args.ref)
    print(f"Reference: {ref_text}")

    asr_model = load_asr_model()
    asr_text, duration = transcribe(asr_model, args.audio)
    print(f"ASR: {asr_text}")
    print(f"Duration: {duration:.1f}s")

    hyp_norm = normalize_text(asr_text)
    print(f"ASR norm: {hyp_norm}")

    ops = align(ref_text, hyp_norm)
    errors = sum(1 for r, h in ops if r != h)
    cer = errors / max(len(ref_text), 1)
    print(f"CER: {cer:.2%} ({errors}/{len(ref_text)})")

    subs = [(r, h) for r, h in ops if r != h]
    if subs:
        print(f"Errors: {subs[:20]}")


if __name__ == "__main__":
    main()
