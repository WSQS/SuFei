"""ASR on A/B test audio files."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_tts import load_asr_model, transcribe, normalize_text, align

AB_DIR = ROOT / "output" / "ab_test"
POEM = "空山不见人，但闻人语响。返景入深林，复照青苔上。"

files = [
    ("A_gt_dur.wav", "A (GT dur)"),
    ("B_pred_dur.wav", "B (pred dur)"),
    ("C_scaled_dur.wav", "C (scaled dur)"),
]

def main():
    print("Loading ASR...")
    asr = load_asr_model()
    ref = normalize_text(POEM)
    print(f"Ref: {ref} ({len(ref)} chars)\n")

    for wav, label in files:
        path = AB_DIR / wav
        if not path.exists():
            print(f"SKIP {label}: not found\n")
            continue

        txt, dur = transcribe(asr, str(path))
        hyp = normalize_text(txt)
        ops = align(ref, hyp)
        errs = sum(1 for r, h in ops if r != h)
        cer = errs / max(len(ref), 1)
        d = sum(1 for r, h in ops if r != '*' and h == '*')
        s = sum(1 for r, h in ops if r != '*' and h != '*' and r != h)
        i = sum(1 for r, h in ops if r == '*' and h != '*')

        print(f"{label} ({dur:.1f}s):")
        print(f"  ASR: {txt}")
        print(f"  CER: {cer:.2%} D={d} S={s} I={i}\n")

if __name__ == "__main__":
    main()
