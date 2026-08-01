# ADR-0009 MAS-based duration learning for data-source independence

- Status: Accepted
- Date: 2026-07-31
- Related: ADR-0008, PROGRESS.md v9 results

## Context

The current FastSpeech 2 student (v9, pred-all CER=24.0%) depends on
teacher (PaddleSpeech FS2 ONNX) internal duration outputs as duration
labels. This creates a hard coupling: clean durations can only come from
the teacher model. Switching to a better teacher (CosyVoice, real human
recordings) requires either running MFA (which produced garbage labels
before — see PROGRESS.md Phase 0) or reverse-engineering the new teacher's
internal duration predictor.

The strategic problem: **duration is the only label that cannot be
extracted directly from (text, audio) pairs**. Mel, F0, and energy are all
computable from audio alone. Duration requires alignment, and alignment has
been the persistent source of data bugs (MFA character matching failure,
1.39x time-axis mismatch, teacher ONNX reverse-engineering).

## Decision

Replace external duration labels with **Monotonic Alignment Search (MAS)**
learned jointly during training, following Glow-TTS / VITS.

Two projection layers (`enc_proj: 256→128`, `mel_proj: 80→128`) map
encoder states and mel targets into a shared space where a cost matrix
is computed. MAS (a dynamic-programming algorithm, O(L×T)) finds the
optimal monotonic alignment at each training step. The resulting hard
durations train the duration predictor; a log-likelihood loss trains the
projection layers to produce alignable representations.

At inference time the alignment module is discarded entirely. The model
is architecturally identical to the current v9: phoneme → encoder →
duration predictor → length regulator → decoder → mel. This preserves
the explicit duration control needed for poetry prosody editing (boundary
tokens, pause insertion) and the ONNX export path.

**Phased rollout:**

- **M1**: Validate MAS alignment quality on the frozen v9 encoder against
  teacher durations (correlation target >0.9).
- **M2**: Train v10 with MAS on the same 300-poem teacher data. Target:
  pred-all CER ≤ 24% (parity with v9).
- **M3**: Unlock new data sources (CosyVoice, human recordings) with
  zero alignment tooling.

## Alternatives considered

- **Keep extracting durations from teacher internals**: Works only for
  PaddleSpeech. Every new teacher requires reverse-engineering its ONNX
  graph. Rejected for long-term maintainability.

- **MFA on new audio**: Already failed twice (character matching bug,
  time-axis mismatch). MFA also requires a pronunciation dictionary and
  adds a heavy preprocessing dependency. Rejected.

- **FastPitch-style soft alignment attention**: Produces soft (probabilistic)
  durations that don't directly map to integer frames for the length
  regulator. Requires an auxiliary attention loss and is harder to control.
  MAS produces hard integer durations natively. Rejected.

- **RAD-TTS-style alignment**: More flexible (non-monotonic) but
  overkill for read poetry where monotonicity is guaranteed. Adds
  implementation complexity with no benefit for this domain. Rejected.

- **VITS end-to-end**: Eliminates the FS2 + vocoder split entirely but
  loses explicit duration control and the ability to swap vocoders.
  Rejected for now; FS2+HiFiGAN architecture stays.

## Results (2026-07-31)

- **M1**: marginal (corr 0.59, frozen encoder + linear projections).
- **M2 achieved** after three recipe iterations (see PROGRESS.md v10a-v10f):
  the working recipe is NeMo-style conv aligner + beta-binomial prior inside
  the softmax (forward-sum CTC + Viterbi path-NLL) + encoder-detached scores
  + an `ema_pct_le2 < 0.20` phase gate before MAS durations drive the mel
  path. v10f + inference dur×1.25: train CER 20.7% (ceiling ~20.2%), holdout
  22.6% vs teacher-labeled v9+scale 19.4% — a 3.2pp premium for full duration
  label independence. Naive Viterbi-only MAS (the original plan) collapsed;
  joint (non-detached) training proved bistable/init-lucky.
- MAS DP vectorized (16.4x); training 24.3 (v9) vs ~10.7 step/s (MAS).
- **M3 unlocked**: any (text, audio) source can now be aligned in-training.
- **M3 Phase 1 validated (2026-07-31)**: cold-start (`--cold_start`, zero
  teacher durations end-to-end) on 320 CosyVoice3-generated poems trained
  cleanly; m3_v1 + dur×1.27 diagnostic hits train CER 19.4% — at the
  CosyVoice direct-read floor (18-20%). Duration-sum ratio 0.789 reproduces
  across all three label sources (0.79/0.80/0.789), confirming the shortfall
  is a training-objective issue (log-L1 bias + clamp(max=100)), not a data
  or alignment issue. Root fix tracked separately (no ×1.27 in production).

## Consequences

- **Gained**: Duration labels from any (text, audio) pair with zero
  external tooling. Data source is fully decoupled from teacher model.
- **Gained**: Eliminates the entire class of alignment/_duration bugs
  (MFA, time-axis, teacher ONNX reverse-engineering).
- **Cost**: +43K parameters (enc_proj + mel_proj), discarded at inference.
  Training cost increases by ~15% due to MAS DP per step.
- **Cost**: Training stability depends on projection layer convergence.
  Warm-starting from v9 encoder mitigates this; M1 validates before M2.
- **Constraint**: ONNX export remains unchanged (alignment module is
  training-only). No Android-side changes needed.
