"""Diag v2: slice-level ASR + cross-version A/B for the poem_0027 doubled-王.

Slices the 唐代 region (4.2-5.25s) and 王维 region (5.25-6.15s) out of each
wav and transcribes them separately, plus full-file ASR of the m3_v1 synths.
"""
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_tts import load_asr_model, transcribe

FILES = [
    ROOT / "data" / "m3_cosyvoice_phase1" / "poem_0027_c0.wav",
    ROOT / "output" / "m3_v6_eval" / "train_poem_0027.wav",
    ROOT / "output" / "m3_v5_eval" / "train_poem_0027.wav",
]
SLICES = [("tangdai", 4.20, 5.25), ("wangwei", 5.25, 6.15),
          ("wide", 4.20, 6.45)]
TMP = ROOT / "logs" / "diag_slices"
TMP.mkdir(parents=True, exist_ok=True)


def read_wav(path):
    with wave.open(str(path), 'rb') as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return x, sr


def write_slice(x, sr, t0, t1, out):
    seg = x[int(t0 * sr):int(t1 * sr)]
    with wave.open(str(out), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(seg.tobytes())


asr = load_asr_model()
out = open(ROOT / "logs" / "diag_0027_v2.txt", "w", encoding="utf-8")
for f in FILES:
    if not f.exists():
        out.write(f"MISSING: {f}\n")
        continue
    tag = f.parent.name + "/" + f.name
    text, _ = transcribe(asr, str(f))
    out.write(f"[full] {tag}: {text}\n")
    x, sr = read_wav(f)
    for name, t0, t1 in SLICES:
        sp = TMP / f"{f.parent.name}_{name}.wav"
        write_slice(x, sr, t0, t1, sp)
        stext, _ = transcribe(asr, str(sp))
        out.write(f"[{name} {t0}-{t1}s] {tag}: {stext}\n")
    out.write("\n")
out.close()
print("DONE")
