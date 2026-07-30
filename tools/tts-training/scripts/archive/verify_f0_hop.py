import pyworld as pw
import numpy as np

sr = 24000
audio = np.random.randn(sr * 3).astype(np.float64) * 0.1

_f0, t = pw.dio(audio, sr, f0_floor=80, f0_ceil=400)

print("dio output: len(f0)=%d, len(t)=%d" % (len(_f0), len(t)))
if len(t) > 1:
    period_sec = t[1] - t[0]
    period_ms = period_sec * 1000
    period_samples = period_sec * sr
    print("frame period: %.6f s = %.1f ms = %.1f samples" % (period_sec, period_ms, period_samples))
    print("Total frames for 3s audio: %d (expected %d at 5ms period)" % (len(_f0), 3000//5))
    print("")
    print("Code uses: f0[start_sample // 256 : end_sample // 256]")
    print("  mel frame i covers samples [i*300, (i+1)*300)")
    print("  code reads f0[i*300//256 : (i+1)*300//256] = f0[i*1.17 : (i+1)*1.17]")
    print("  But F0 has period %.1f samples, so correct would be f0[i*300//%.0f : ...]" % (period_samples, period_samples))
    print("")
    print("  BUG: code assumes F0 hop=256 samples, actual=%.0f samples" % period_samples)
    print("  For mel of 100 frames (30s), code reads f0[0:117], but only first %d%% of F0 curve used" % int(100*period_samples/300))
