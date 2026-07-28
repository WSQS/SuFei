"""Batch ASR analysis of generated audio files."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_tts import load_asr_model, transcribe, normalize_text, align
from nar_build_manifest import text_to_phonemes

NAR_DIR = ROOT / "output" / "nar_inference"

TEST_CASES = [
    ("demo_chunxiao.wav", "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"),
    ("demo_jingyesi.wav", "床前明月光，疑是地上霜。举头望明月，低头思故乡。"),
    ("demo_wangyue.wav", "岱宗夫如何？齐鲁青未了。造化钟神秀，阴阳割昏晓。"),
]

PUNCT = set("，。？！；：·")


def compute_per(ref_phones, hyp_text, phone_map):
    """Compute PER using G2P alignment."""
    hyp_phones = []
    for ph, _ in text_to_phonemes(hyp_text):
        if ph not in {"<eos>"}:
            hyp_phones.append(ph)
    ref_phones_filtered = [p for p in ref_phones if p not in PUNCT]

    ops = align(ref_phones_filtered, hyp_phones)
    errors = sum(1 for r, h in ops if r != h)
    return errors / max(len(ref_phones_filtered), 1), errors, len(ref_phones_filtered)


def main():
    print("Loading ASR model...")
    asr = load_asr_model()
    print("Ready.\n")

    # Load phone map for PER
    phone_map = {}
    pmap_path = ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0" / "phone_id_map.txt"
    with open(pmap_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                phone_map[parts[0]] = int(parts[1])

    results = []
    for wav_name, ref_text in TEST_CASES:
        wav_path = NAR_DIR / wav_name
        if not wav_path.exists():
            print(f"SKIP {wav_name}: not found")
            continue

        print(f"\n=== {wav_name} ===")

        asr_text, duration = transcribe(asr, str(wav_path))
        print(f"  ASR: {asr_text}")

        ref_norm = normalize_text(ref_text)
        hyp_norm = normalize_text(asr_text)

        # CER
        ops = align(ref_norm, hyp_norm)
        cer_errors = sum(1 for r, h in ops if r != h)
        cer = cer_errors / max(len(ref_norm), 1)

        # PER
        ref_phones = [ph for ph, _ in text_to_phonemes(ref_text) if ph not in {"<eos>"} and ph not in PUNCT]
        hyp_phones_raw = [ph for ph, _ in text_to_phonemes(asr_text) if ph not in {"<eos>"} and ph not in PUNCT]
        p_ops = align(ref_phones, hyp_phones_raw)
        per_errors = sum(1 for r, h in p_ops if r != h)
        per = per_errors / max(len(ref_phones), 1)

        print(f"  Ref (norm): {ref_norm} ({len(ref_norm)} chars)")
        print(f"  ASR (norm): {hyp_norm} ({len(hyp_norm)} chars)")
        print(f"  CER: {cer:.2%} ({cer_errors}/{len(ref_norm)})")
        print(f"  PER: {per:.2%} ({per_errors}/{len(ref_phones)})")
        print(f"  Duration: {duration:.1f}s")

        # Show substitutions
        subs = [(r, h) for r, h in ops if r != h]
        if subs:
            print(f"  CER errors: {subs[:15]}")

        results.append({
            "file": wav_name, "text": ref_text,
            "asr": asr_text, "duration": duration,
            "cer": round(cer, 4), "cer_errors": cer_errors,
            "per": round(per, 4), "per_errors": per_errors,
            "ref_chars": len(ref_norm),
        })

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['file']}: CER={r['cer']:.2%} PER={r['per']:.2%} dur={r['duration']:.1f}s")

    if results:
        avg_cer = sum(r["cer"] for r in results) / len(results)
        avg_per = sum(r["per"] for r in results) / len(results)
        print(f"\n  Avg CER: {avg_cer:.2%}")
        print(f"  Avg PER: {avg_per:.2%}")


if __name__ == "__main__":
    main()
