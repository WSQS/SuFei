"""ASR analysis of overfit test results."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_tts import load_asr_model, transcribe, normalize_text, align

OVERFIT_DIR = ROOT / "output" / "overfit"
POEM_TEXT = "空山不见人，但闻人语响。返景入深林，复照青苔上。"

FILES = [
    ("poem_0285_audio_teacher.wav", "Teacher"),
    ("poem_0285_audio_gt.wav", "GT mel vocoded"),
    ("poem_0285_audio_pred.wav", "Overfit pred"),
]


def main():
    print("Loading ASR model...")
    asr = load_asr_model()
    print("Ready.\n")

    ref_norm = normalize_text(POEM_TEXT)
    print(f"Reference: {ref_norm} ({len(ref_norm)} chars)\n")

    for wav_name, label in FILES:
        wav_path = OVERFIT_DIR / wav_name
        if not wav_path.exists():
            print(f"SKIP {label}: {wav_name} not found\n")
            continue

        asr_text, duration = transcribe(asr, str(wav_path))
        hyp_norm = normalize_text(asr_text)

        ops = align(ref_norm, hyp_norm)
        errors = sum(1 for r, h in ops if r != h)
        cer = errors / max(len(ref_norm), 1)

        # Count D/S/I
        deletions = sum(1 for r, h in ops if r != '*' and h == '*')
        substitutions = sum(1 for r, h in ops if r != '*' and h != '*' and r != h)
        insertions = sum(1 for r, h in ops if r == '*' and h != '*')

        print(f"=== {label} ===")
        print(f"  ASR: {asr_text}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  CER: {cer:.2%} ({errors}/{len(ref_norm)})")
        print(f"  D={deletions} S={substitutions} I={insertions}")
        print()


if __name__ == "__main__":
    main()