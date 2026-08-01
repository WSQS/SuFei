"""One-off diag: ASR transcripts of poem_0027 source candidates + m3_v4 synth."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_tts import load_asr_model, transcribe

REF = ("送元二使安西 / 渭城曲，唐代·王维。渭城朝雨浥轻尘，客舍青青柳色新。"
       "劝君更尽一杯酒，西出阳关无故人。")

FILES = [
    ROOT / "data" / "m3_cosyvoice_phase1" / "poem_0027_c0.wav",
    ROOT / "data" / "m3_cosyvoice_phase1" / "poem_0027_c1.wav",
    ROOT / "output" / "m3_v4_eval" / "train_poem_0027.wav",
]

asr = load_asr_model()
out = open(ROOT / "logs" / "diag_0027.txt", "w", encoding="utf-8")
out.write(f"REF: {REF}\n")
for f in FILES:
    if not f.exists():
        out.write(f"MISSING: {f}\n")
        continue
    text, _ = transcribe(asr, str(f))
    out.write(f"{f.name}: {text}\n")
out.close()
