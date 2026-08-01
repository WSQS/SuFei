"""Zoom analysis 4.2-7.6s: syllable-level RMS + voicing character per burst."""
import sys
import wave
import numpy as np


def load(path):
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()),
                          dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def band_ratio(seg, sr):
    """low(80-1000Hz) vs high(2000-8000Hz) energy in dB; voiced syllables
    are strongly low-dominant, breaths are not."""
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    low = spec[(freqs >= 80) & (freqs < 1000)].sum()
    high = spec[(freqs >= 2000) & (freqs < 8000)].sum()
    return 10 * np.log10((low + 1e-12) / (high + 1e-12))


def f0_autocorr(seg, sr):
    """crude F0 via autocorrelation peak in 60-400 Hz; 0 if unvoiced."""
    seg = seg - seg.mean()
    if (seg ** 2).mean() < 1e-8:
        return 0.0
    ac = np.correlate(seg, seg, 'full')[len(seg) - 1:]
    ac /= (ac[0] + 1e-12)
    lo, hi = int(sr / 400), int(sr / 60)
    if hi >= len(ac):
        return 0.0
    k = lo + np.argmax(ac[lo:hi])
    return sr / k if ac[k] > 0.35 else 0.0


for path in sys.argv[1:]:
    x, sr = load(path)
    hop = int(0.005 * sr)
    win = int(0.020 * sr)
    print(f"== {path}")
    t = 4.2
    rows = []
    while t < 7.6:
        i = int(t * sr)
        seg = x[i:i + win]
        if len(seg) < win:
            break
        rms_db = 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-6)
        rows.append((t, rms_db))
        t += 0.005
    # print compact envelope: one char per 20ms (4 rows)
    print("   envelope (each char=20ms, . <-62dB, - <-50, x <-40, X >=-40):")
    line = []
    for j in range(0, len(rows), 4):
        db = max(r[1] for r in rows[j:j + 4])
        line.append('.' if db < -62 else '-' if db < -50 else
                    'x' if db < -40 else 'X')
    s = ''.join(line)
    for k in range(0, len(s), 85):
        t0 = 4.2 + k * 0.02
        print(f"   {t0:5.2f}s  {s[k:k + 85]}")
    # per-window voicing details on a coarser 40ms grid
    print("   t(s)   rms_dB  low/high_dB  F0(Hz)")
    t = 4.2
    while t < 7.6:
        i = int(t * sr)
        seg = x[i:i + int(0.040 * sr)]
        if len(seg) < int(0.040 * sr):
            break
        rms_db = 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-6)
        if rms_db > -55:
            br = band_ratio(seg, sr)
            f0 = f0_autocorr(seg, sr)
            print(f"   {t:5.2f}  {rms_db:6.1f}  {br:8.1f}    "
                  f"{f0:6.0f}" if f0 else
                  f"   {t:5.2f}  {rms_db:6.1f}  {br:8.1f}       -")
        t += 0.040
