# ADR-0008 From-scratch FS2 training: generalization ceiling

- Status: Accepted
- Date: 2026-07-29
- Related: ADR-0006, ADR-0007, `feat/tts-training-data` branch

## Context

After ADR-0007 settled on PaddleSpeech FS2 ONNX as the production TTS,
a parallel research track explored training a compact 7.6M FastSpeech 2
from scratch on PaddleSpeech-generated distillation data, aiming for a
smaller on-device model with better poetry-specific prosody.

Three controlled experiments were conducted to isolate the bottleneck:

### H1: Baseline (171 poems, full-poem training)

- Train: 171 poems, GT variance mode, decoder mask, 24k steps
- Train delta: +1.7% (mel L1 = 0.130)
- Holdout-20 delta: +70.8% (mel L1 = 0.569)
- Holdout P3=100% (complete ASR failure): 11/20
- Step curve (12k→18k→24k): delta flat at 69.7→71.1→70.8%, no improvement

**Conclusion**: The model memorizes training sequences; it does not learn
composable phoneme→acoustic mappings.

### H2: Mixed-window training (same 171 poems)

- 50% full poem, 25% single line, 25% couplet (dynamic per-sample)
- Same model, LR, mask, loss, seed as H1
- Holdout-20 delta: +69.9% (down <1pp from H1)
- Line-level delta: +0.1% — but P3=100% = 126/126 (all lines fail)
- Window ratio verified: full=49.9%, line=25.0%, couplet=25.0%

**Conclusion**: Training unit (full poem vs mixed window) is not the
bottleneck.

### D300: Data scaling (300 poems, full-poem training)

- Nested: train_171 ⊂ train_300, same holdout_20
- Holdout-20 delta: +69.9% at 24k (down <1pp from H1)
- Early checkpoint curve (500→3000 steps): delta 72.8%→70.2%, no U-shape
- Val mel L1 curve: lowest at step ~800 (0.538), rebounds to 0.555 at 6k,
  settles at 0.545 at 24k — but ASR delta shows no corresponding U-shape

**Conclusion**: Data scaling 171→300 is ineffective. The val mel L1
U-shape is misleading — it reflects initial descent, not a generalization
window.

### Summary table

| Experiment | Data | Training | Holdout delta | Train delta |
|------------|------|----------|--------------|-------------|
| H1 | 171 | Full poem | +70.8% | +1.7% |
| H2 | 171 | Mixed window | +69.9% | — |
| D300 | 300 | Full poem | +69.9% | — |

## Decision

The from-scratch 7.6M FS2 training track is **suspended**. The model
architecture (7.6M, d_model=256) combined with available data scale
(171–300 poems) cannot achieve generalization. The gap between train
(+1.7%) and holdout (+70.8%) is a structural failure, not a tuning issue.

Production continues with PaddleSpeech FS2 ONNX (ADR-0007). Future
on-device quality improvement should pursue one of:

1. **Pre-trained initialization**: warm-start from PaddleSpeech weights
   (37.3M, d_model=384) rather than training from scratch
2. **Better teacher**: use CosyVoice 3 (already deployed in WSL) to
   generate higher-quality poetry audio, then fine-tune
3. **Accept on-device limitations**: optimize only for poems in the
   training set (no holdout generalization required)

## Alternatives considered

- **More data (600+ poems)**: D300 showed ~1pp improvement over D171.
  Extrapolation suggests 600 would yield diminishing returns. Not worth
  the annotation cost without an architectural change.
- **Different architecture**: the 7.6M FS2 is already minimal. A larger
  model (matching PaddleSpeech 37.3M) would face the same from-scratch
  problem with more parameters to overfit.
- **Stronger regularization**: dropout, weight decay, and early stopping
  were considered. The D300 early-checkpoint experiment (step 500-3000)
  showed no generalization window — regularization cannot preserve what
  was never learned.

## Consequences

- PaddleSpeech FS2 ONNX remains the only production TTS path
- All H1/H2/D300 checkpoints, manifests, and evaluation scripts are
  preserved for future analysis but are not production candidates
- The comma G2P bug (pid=263 mapped to `<unk>`) was fixed; future
  experiments will use corrected manifests
- The validation mel L1 logging (`--val_manifest`, `--val_interval`)
  added to `nar_train.py` remains available for future training
