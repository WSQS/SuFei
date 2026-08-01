"""M3 Phase 0: calibrate the mel extraction convention against HiFiGAN's
true input space.

Self-contained round trip on rtx:
  mel_native (teacher FS2 dump, from paddle_distill_features_v3 npz)
    -> HiFiGAN -> audio
    -> extract with candidate conventions (power 1/2 x ln/log10)
    -> compare to mel_native (raw L1 + best-affine-fit L1)

The convention with the smallest RAW L1 is what HiFiGAN was trained on and
is the one M3 must use when extracting mel from new audio sources
(CosyVoice / human recordings). The librosa power=2 + ln convention used by
the old extract_mel() is expected to lose badly here (it caused the 31.8%
ceiling).
"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa
import onnxruntime as ort

ROOT = Path('E:/sufei-training')
FEAT_DIR = ROOT / 'data' / 'paddle_distill_features_v3'
HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0' / 'hifigan_csmsc.onnx'

SR = 24000
N_FFT = 2048
HOP = 300
WIN = 1200
N_MELS = 80
FMIN = 80
FMAX = 7600

MEL_BASIS = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)


def extract(audio, power, use_log10, eps):
    S = np.abs(librosa.stft(y=audio, n_fft=N_FFT, hop_length=HOP,
                            win_length=WIN, window='hann')) ** power
    mel = np.maximum(np.dot(MEL_BASIS, S), eps)
    logmel = np.log10(mel) if use_log10 else np.log(mel)
    return logmel.T.astype(np.float32)  # (T, 80)


def main():
    voc = ort.InferenceSession(str(HIFIGAN), providers=['CPUExecutionProvider'])
    npzs = sorted(FEAT_DIR.glob('*.npz'))[:3]
    print(f"Calibrating on {len(npzs)} samples: {[p.stem for p in npzs]}")

    variants = [
        ('power2_ln_1e-5', 2, False, 1e-5),   # the old extract_mel() convention
        ('power1_ln_1e-5', 1, False, 1e-5),
        ('power2_log10_1e-10', 2, True, 1e-10),
        ('power1_log10_1e-10', 1, True, 1e-10),
        ('power1_ln_1e-10', 1, False, 1e-10),
    ]

    agg = {name: {'raw': [], 'affine': [], 'a': [], 'b': []} for name, *_ in variants}

    for npz_path in npzs:
        mel_native = np.load(npz_path)['mel'].astype(np.float32)  # (T, 80)
        audio = voc.run(None, {'logmel': mel_native})[0].flatten()

        for name, power, use_log10, eps in variants:
            mel_x = extract(audio, power, use_log10, eps)
            T = min(len(mel_x), len(mel_native))
            x, y = mel_x[:T].ravel(), mel_native[:T].ravel()
            raw_l1 = float(np.mean(np.abs(x - y)))
            # best affine fit y ~ a*x + b
            a, b = np.polyfit(x, y, 1)
            affine_l1 = float(np.mean(np.abs(a * x + b - y)))
            agg[name]['raw'].append(raw_l1)
            agg[name]['affine'].append(affine_l1)
            agg[name]['a'].append(float(a))
            agg[name]['b'].append(float(b))

    print(f"\n{'variant':24s} {'raw_L1':>8s} {'affine_L1':>10s} {'a':>7s} {'b':>8s}")
    best = None
    for name, *_ in variants:
        r = np.mean(agg[name]['raw'])
        af = np.mean(agg[name]['affine'])
        a = np.mean(agg[name]['a'])
        b = np.mean(agg[name]['b'])
        print(f"{name:24s} {r:8.4f} {af:10.4f} {a:7.3f} {b:8.3f}")
        if best is None or r < best[1]:
            best = (name, r)

    print(f"\nWINNER (raw L1): {best[0]}  L1={best[1]:.4f}")
    print("If winner raw_L1 << others and a~1, b~0: that IS the HiFiGAN convention.")


if __name__ == '__main__':
    main()
