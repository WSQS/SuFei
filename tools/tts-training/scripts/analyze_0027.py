"""One-off: burst analysis of the 唐代·王维。 region in poem_0027 wavs."""
import sys
import wave
import numpy as np


def load(path):
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)
    return x / 32768.0, sr


def analyze(path, t_end=9.0):
    x, sr = load(path)
    hop, win = int(0.010 * sr), int(0.025 * sr)
    n_frames = (len(x) - win) // hop
    rms = np.array([np.sqrt((x[i * hop:i * hop + win] ** 2).mean())
                    for i in range(n_frames)])
    db = 20 * np.log10(rms + 1e-6)
    floor = np.percentile(db, 10)
    peak = np.percentile(db, 95)
    thr = floor + 0.35 * (peak - floor)
    voiced = db > thr

    # segments of continuous voiced/silence, in seconds
    segs = []
    cur, start = voiced[0], 0
    for i in range(1, n_frames):
        if voiced[i] != cur:
            segs.append((cur, start * 0.010, i * 0.010))
            cur, start = voiced[i], i
    segs.append((cur, start * 0.010, n_frames * 0.010))

    print(f"== {path}  (sr={sr}, dur={len(x)/sr:.2f}s, thr={thr:.1f}dB)")
    for v, s, e in segs:
        if e > t_end:
            break
        tag = 'SOUND' if v else 'sil  '
        bar = '#' * min(60, int((e - s) * 50))
        print(f"  {tag} {s:6.2f}-{e:6.2f}  ({e-s:5.2f}s) {bar}")


for p in sys.argv[1:]:
    analyze(p)
