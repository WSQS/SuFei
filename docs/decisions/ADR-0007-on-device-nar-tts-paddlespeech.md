# ADR-0007 On-device NAR TTS via PaddleSpeech FS2 + HiFi-GAN ONNX

- Status: Accepted
- Date: 2026-07-28
- Related: ADR-0006, `feat/tts-training-data` branch

## Context

The fork requires on-device Chinese poetry TTS. Three approaches were
explored (documented in ADR-0006):

1. **System TTS** (PR #34): baseline, device-dependent
2. **MOSS-TTS-Nano** (AR): archived as NO-GO (token absorbing states)
3. **Custom FastSpeech 2** (NAR): trained from scratch on 66 min of
   CosyVoice 3 teacher audio

The custom FS2 training revealed that 66 minutes is insufficient for
a full FS2 model. Key findings from the training pipeline:

- Duration predictor required log1p transform (not log+clamp)
- Acoustic decoder generalizes poorly beyond training samples
- Even with GT variance, only 25% of samples reach CER<15%

## Decision

Use **PaddleSpeech pre-trained FS2 ONNX + HiFi-GAN ONNX** as the
production TTS engine, with the custom-trained model as a future
fine-tuning path.

### Rationale

PaddleSpeech FS2 was trained on 12 hours of CSMSC data:

| Metric | PaddleSpeech | Custom FS2 |
|--------|-------------|------------|
| Test CER | 6.8% | 45.8% |
| Pass (<15%) | 21/24 | 3/12 |
| RTF (PC) | 0.011 | 0.010 |
| Duration ratio | ~100% | ~77% |

The pre-trained model works out-of-the-box on Chinese poetry without
fine-tuning, achieving CER=6.8% on the 24-poem test set.

### Android integration

```
Text → ChineseG2p (Kotlin) → phone IDs (int64)
  → FS2 ONNX (142MB) → mel [T,80]
  → HiFi-GAN ONNX (50MB) → waveform [N]
  → AudioTrack (24kHz, mono, PCM_FLOAT)
```

Model files are NOT bundled in the APK (192MB total). Users push
them to `getExternalFilesDir/null/models/nar/` via ADB.

### G2P

A Kotlin port of the Python `text_to_phones` function:
- TinyPinyin for character→pinyin lookup
- Tone mark→number conversion (chūn → chun1)
- Pseudo-initial handling (y/w → i/u)
- Final remapping (un→uen, ui→uei, etc.)
- i/ii/iii disambiguation by initial consonant

### Custom training artifacts

The training pipeline produced valuable diagnostic tools and findings:
- log1p transform for duration predictor (breaks 30% ratio ceiling)
- A/B isolation matrix for variance predictor evaluation
- Acoustic baseline checkpoint (step12000, GT variance CER=0%)

These remain available for future fine-tuning on PaddleSpeech weights.

## Consequences

- **Production-ready**: CER=6.8%, deterministic, zero early-stop/repeat
- **No training needed**: pre-trained model works immediately
- **192MB external**: models must be provisioned separately from APK
- **Future fine-tuning**: can warm-start from PaddleSpeech weights
  instead of training from scratch
- **G2P port**: Kotlin implementation must stay in sync with Python
  reference for phoneme vocabulary compatibility
