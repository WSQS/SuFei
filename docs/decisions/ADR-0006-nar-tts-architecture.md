# ADR-0006 NAR TTS architecture (FastSpeech 2 + HiFi-GAN)

- Status: Accepted
- Date: 2026-07-27
- Related: `feat/tts-training-data` branch, NAR_INTERFACE_SPEC.md

## Context

The fork needs on-device Chinese poetry TTS. Three architectures were
explored:

1. **System TTS** (PR #34): baseline quality, zero deployment cost, but
   no prosody control and device-dependent.
2. **MOSS-TTS-Nano** (AR, PR #38): autoregressive token prediction.
   Archived as NO-GO after v4 diagnostics: greedy decoding enters token
   absorbing states (0/24 pass), and teacher-forced accuracy was ~0.5%
   due to likely logit-ID space mismatch.
3. **NAR (FastSpeech 2 + HiFi-GAN)**: non-autoregressive, deterministic,
   feed-forward. Adopted as product mainline.

The NAR approach was chosen because it is architecturally incapable of
the failure modes that killed MOSS-TTS-Nano: no autoregressive loop
means no token absorbing, no early stop, and bitwise-deterministic
output by construction.

## Decision

Adopt FastSpeech 2 (acoustic model) + HiFi-GAN (vocoder) as the NAR
TTS architecture. The full stack:

```
Text → pypinyin G2P → PaddleSpeech phone IDs
  → FastSpeech 2 (duration + pitch + energy + mel)
  → HiFi-GAN (mel → waveform)
```

### Phoneme inventory

Reuse the PaddleSpeech CSMSC phone vocabulary (268 entries). This gives
us a pre-trained HiFi-GAN vocoder compatible with our mel format, and
avoids training a vocoder from scratch. The vocabulary covers all
initials/finals/tones needed for Tang poetry.

### Mel parameters

Frozen in `NAR_INTERFACE_SPEC.md` v1.0:
- sr=24000, hop=300, n_fft=2048, win=1200, n_mels=80
- fmin=80, fmax=7600 (matching PaddleSpeech FS2 CSMSC defaults)

### Duration source

MFA forced alignment on teacher audio (CosyVoice 3). Per-character
durations extracted from TextGrid word tiers, split into initial/final
(30/70 ratio). Punctuation silence allocated from TextGrid empty
intervals.

Invariant: `sum(per-phoneme durations) == mel_length` for all training
samples (enforced by manifest builder).

### Training data

- 320 CosyVoice 3 teacher audio generated (all Tang 300)
- 221 passed ASR quality filter (67.1 min)
- 219 MFA-aligned (2 long poems dropped)
- 204 used for training (mel_len ≤ 2000)
- Mel/F0/energy extracted per-sample (.npz)

### Model architecture

Minimal FastSpeech 2 (7.6M params):
- Phoneme embedding (268 → 256)
- 4× Transformer encoder (d=256, h=2, ff=1024)
- 3× Variance predictor (duration, pitch, energy)
- Length regulator
- 4× Transformer decoder
- Linear mel head (256 → 80)

## Consequences

- **Deterministic inference**: same input always produces identical
  output. Satisfies the bitwise-stability hard requirement.
- **No autoregressive failures**: structurally impossible to have token
  absorbing, early stop, or line repetition.
- **ONNX-exportable**: both FastSpeech 2 and HiFi-GAN export as ONNX
  with dynamic sequence axes for Android deployment.
- **Duration predictor challenge**: unlike AR models, NAR must learn
  duration explicitly. Overfit tests show mel loss converges fast but
  duration prediction needs careful loss balancing and sufficient
  training data.
- **Pre-trained vocoder reuse**: HiFi-GAN CSMSC vocoder was trained on
  female speech. CosyVoice 3 teacher audio has a different voice
  character. This mismatch may limit output quality until a custom
  vocoder is fine-tuned on our mel domain.
