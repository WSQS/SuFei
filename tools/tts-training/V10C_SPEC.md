# v10c Spec: MAS stabilization (beta-binomial prior + forward-sum loss + MAS-dur expansion)

## Background / diagnosis

- **v10a collapse**: bare Viterbi-only alignment loss let MAS drift to skewed
  alignments (most phonemes squeezed to 1-2 frames, silences absorbing hundreds).
  The duration predictor then learned the short distribution -> inference total
  duration collapsed to ~20% of target.
- **v10b plateau**: Phase 2 expands the decoder input with *predicted* durations,
  so mel/pitch/energy losses compare frame-misaligned sequences. mel loss
  plateaus at ~0.25 (v9 level is ~0.15); pitch/energy inflate from 0.33/0.42 to
  0.60/0.65. This is the known "full_e2e misaligned loss" failure mode.
- **v10c fix** (this spec): stabilize MAS itself with the standard RAD-TTS /
  FastPitch aligner recipe — a static beta-binomial diagonal prior guiding the
  DP search plus a CTC forward-sum loss over *all* monotonic paths — and expand
  with **MAS durations** in Phase 2. MAS durations sum exactly to T, so every
  loss stays frame-aligned; the collapse is prevented by the prior + forward-sum.

## Changes to `scripts/mas.py`

### 1. Module-level prior function

```python
from functools import lru_cache
import math

@lru_cache(maxsize=4096)
def log_beta_binomial_prior(L, T, scaling=1.0):
    """Log beta-binomial diagonal prior, shape [L, T] float32 numpy array.

    For mel frame t (0-indexed, t in [0, T)), the prior over phoneme index
    k in [0, L) is BetaBinomial(n=L-1, a=scaling*(t+1), b=scaling*(T-t)).

    log_pmf(k; n, a, b) = logC(n, k) + betaln(k + a, n - k + b) - betaln(a, b)
    where betaln(x, y) = lgamma(x) + lgamma(y) - lgamma(x + y)
    and   logC(n, k)   = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)

    Pure numpy + math.lgamma — do NOT import scipy.
    Callers must treat the returned (cached) array as read-only.
    """
```

Vectorize over k (numpy `scipy`-free `lgamma` via `np.vectorize(math.lgamma)`
or `scipy`-free gammaln equivalent; a Python loop over T with vectorized k is
acceptable — the lru_cache means each (L, T) pair is computed once).

### 2. `ForwardSumLoss` (NeMo / FastPitch style)

```python
class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1.0): ...

    def forward(self, attn_logits, phone_lens, mel_lens):
        """attn_logits: [B, L, T] RAW (pre-softmax) alignment scores.
        phone_lens, mel_lens: [B] ints. Returns scalar loss (batch mean)."""
```

Per sample `b` (Python loop over B is fine, B=8):

1. `s = attn_logits[b, :Lb, :Tb].transpose(0, 1)` -> `[Tb, Lb]`
2. Pad a blank column at class index 0 filled with `blank_logprob` -> `[Tb, Lb+1]`
3. `log_softmax` over last dim, reshape to `[Tb, 1, Lb+1]`
4. `targets = torch.arange(1, Lb + 1, device=..., dtype=torch.long).unsqueeze(0)`
5. `loss_b = F.ctc_loss(logprob, targets, input_lengths=(Tb,), target_lengths=(Lb,), blank=0, zero_infinity=True)`

Return mean over batch.

### 3. `AlignmentModule.compute_scores`

```python
def compute_scores(self, enc_out, mel_target):
    """Raw scaled dot-product scores [B, L, T] (no softmax, no masking)."""
```

Refactor `compute_log_prob` to call `compute_scores` internally and then apply
`log_softmax(dim=1)` + mel-padding `masked_fill` exactly as today. Its public
signature and behavior must NOT change (`eval_mas_alignment.py` depends on it).

## Changes to `scripts/nar_train_align.py`

1. Import `ForwardSumLoss`, `log_beta_binomial_prior` from `mas`.
2. New CLI args: `--w_fsum` (float, default 1.0), `--prior_scale` (float,
   default 1.0), `--prior_off` (store_true — disables the prior).
3. Instantiate `fsum_loss_fn = ForwardSumLoss()` next to the align module.
4. Training loop, MAS block rework:
   - `scores = align_module.compute_scores(x, mel_norm)` once per step;
     derive `log_prob` from scores exactly as `compute_log_prob` does
     (log_softmax over dim=1, then masked_fill of padded mel frames), so scores
     and log_prob stay consistent.
   - Per-sample DP: build the cost in numpy as
     `cost = log_prob[b, :L, :T].detach().cpu().numpy() + log_beta_binomial_prior(L, T, args.prior_scale)`
     (skip the prior term when `--prior_off`). The prior guides ONLY the DP
     search; the alignment losses are computed on the un-prior'd model scores.
   - `fsum = fsum_loss_fn(scores, phone_lens, mel_lens)` (per-sample slicing
     inside the loss makes masking unnecessary).
5. **Phase 2 expansion**: `dur_for_expand = mas_durations` (delete the
   pred-dur expansion). Phase 1 stays `durations_gt` (teacher warmup).
6. `total_loss += args.w_fsum * fsum`; log as `fsum=` in the console line and
   `loss/fsum` in wandb.
7. Diagnostics next to dur_corr/dur_ratio: `pct_le2` = fraction of MAS
   durations <= 2 among valid phones in the batch; log + wandb
   `metrics/pct_dur_le2`.
8. Wrap `wandb.init(...)` in try/except; on exception retry with
   `mode="offline"` and print a warning (fixes silent startup crash risk).
9. Nothing else changes: dataset, model, optimizer, checkpoint format, and the
   `clamp(min=1, max=100)` duration target all stay as-is.

## New file `scripts/test_v10c_smoke.py`

Synthetic-tensor smoke test (CPU, no data files):

- (a) `log_beta_binomial_prior(20, 200)`: shape [20, 200], all finite,
  per-frame argmax is non-decreasing (roughly diagonal).
- (b) `ForwardSumLoss`: finite scalar on random `[2, L, T]` logits with
  `phone_lens=[5, 7]`, `mel_lens=[40, 55]`; after ~50 Adam steps optimizing a
  small projection to a synthetic block-diagonal target alignment, the loss
  decreases.
- (c) MAS + prior: build a noisy cost matrix whose true alignment is a known
  block-diagonal path; check recovered durations match ground truth within
  +/-2 frames per phoneme.

Print PASS/FAIL per section, exit non-zero on failure.

## Acceptance

- `python -m py_compile scripts/mas.py scripts/nar_train_align.py` passes.
- `python scripts/test_v10c_smoke.py` passes all three sections.
- Only `scripts/mas.py`, `scripts/nar_train_align.py`,
  `scripts/test_v10c_smoke.py` are modified/created. No reformatting of
  unrelated code. Do not run training.

---

# v10d Addendum: aligner capacity + prior-in-softmax + phase gate

## Diagnosis from v10c runs

Three v10c runs, three outcomes (fsum plateau 2.6 / 7.5 / 13) — the aligner
optimization is bistable and luck-dominated:

- **Joint training** (smoke, j240): fsum gradient reshapes the encoder, which
  provides the capacity the aligner needs (fast convergence when lucky) but
  wrecks the warm-started v9 encoder (mel 0.12→0.44) and is init-sensitive.
- **Detached** (j241): encoder protected (mel stable 0.17 ✓) but the two bare
  Linear projections lack capacity to learn alignment on frozen features —
  fsum stuck at 12-14, pct_le2 0.88.

Two deviations from the proven NeMo/FastPitch AlignmentEncoder recipe are
responsible:

1. Their aligner has its own **conv stacks + L2-distance scores** (capacity
   lives in the aligner, not the encoder).
2. The **beta-binomial prior is added to the attention logits INSIDE the
   softmax used by the forward-sum loss** — our fsum saw raw skewed scores,
   so CTC locked onto the skewed basin. Prior-in-DP-only is not enough.

## Changes to `scripts/mas.py`

### 1. AlignmentModule → NeMo-style conv architecture

Keep the class name and constructor signature
`AlignmentModule(d_model=256, n_mels=80, d_align=128)`. Replace the two
Linear layers with (NeMo AlignmentEncoder shape):

```python
self.key_proj = nn.Sequential(            # text branch: [B, L, d_model]
    nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv1d(d_model * 2, d_align, kernel_size=1),
)
self.query_proj = nn.Sequential(          # mel branch: [B, T, n_mels]
    nn.Conv1d(n_mels, n_mels * 2, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv1d(n_mels * 2, n_mels, kernel_size=1), nn.ReLU(),
    nn.Conv1d(n_mels, d_align, kernel_size=1),
)
self.temperature = 0.0005
```

`compute_scores(enc_out, mel_target)` (same signature/return as now):
transpose to channel-first for the convs, then

```python
# keys: [B, L, D], queries: [B, T, D]
# scores[b, l, t] = -temperature * ||keys[b, l] - queries[b, t]||^2
scores = -self.temperature * (
    (keys.unsqueeze(2) - queries.unsqueeze(1)) ** 2
).sum(-1)                                  # [B, L, T]
```

(L×T×D memory: 8×300×2000×128 floats is too large — compute via the expanded
form `||k||² + ||q||² - 2 k·q` with bmm instead, which is [B, L, T] only.)

`compute_log_prob` behavior unchanged (softmax over dim=1 + mel mask).

### 2. ForwardSumLoss: optional prior

`forward(self, attn_logits, phone_lens, mel_lens, prior=None)` — if `prior`
(a [B, L_max, T_max] log-prior tensor) is given, use
`attn_logits = attn_logits + prior` before the per-sample slicing. Everything
else unchanged.

## Changes to `scripts/nar_train_align.py`

1. Each step, assemble a padded per-batch log-prior tensor
   `prior_t [B, L_max, T_max]` (zeros in padded regions) from the cached
   `log_beta_binomial_prior(L, T, args.prior_scale)` per sample; skip entirely
   when `--prior_off`.
2. `log_prob = F.log_softmax(scores + prior_t, dim=1)` then the existing
   mel-mask fill — the prior now shapes BOTH the path-NLL and the DP cost.
   **Remove the numpy prior addition inside the per-sample DP loop**
   (no double-counting).
3. `fsum = fsum_loss_fn(scores, phone_lens, mel_lens, prior=prior_t)`.
4. **Phase gate**: maintain `ema_pct_le2` (init 1.0, `ema = 0.98*ema +
   0.02*batch_pct_le2`, updated EVERY step — compute batch pct_le2 every step,
   it is cheap). Phase 2 condition becomes
   `use_mas = step >= args.align_warmup and ema_pct_le2 < 0.20`.
   Print one line the first time the gate opens; log `metrics/ema_pct_le2`
   to wandb. If the gate never opens the run stays in warmup mode (safe).
5. Keep: `x.detach()` for scores, seed handling, all other v10c behavior.

## test_v10c_smoke.py updates

- Section (b): construct the new AlignmentModule, check `compute_scores`
  output shape [B, L, T] and finiteness; test ForwardSumLoss both with
  `prior=None` and with a batched prior tensor (finite, decreases over ~50
  Adam steps on the module's own parameters using synthetic diagonal data).
- Section (c) unchanged.
- Same acceptance: py_compile + all sections PASS.

---

# v10e Addendum: MAS training speedup (no dependency changes)

## Motivation

v9 trains at 24.3 step/s; v10d at 3.2 step/s (7.6x slower, 24k steps = 2h05m
vs 16m). Dominant cost: `maximum_path_np` is a pure-Python per-cell double
loop (L×T ≈ 80k Python iterations per sample, ×8 per step), plus 8 separate
GPU→CPU syncs per step in the DP loop. Training-only cost — inference model
unaffected. Target: 12+ step/s.

## Changes to `scripts/mas.py`

Vectorize the DP in `maximum_path_np` (keep the same function name and
signature; keep the T<L padding behavior):

- Column sweep over t = 1..T-1; vectorize over the phoneme dimension:
  `shifted[i] = dp_prev[i-1]` (prepend -1e9), `trans_wins = shifted >= dp_prev`,
  `dp_cur = cost[:, t] + np.maximum(dp_prev, shifted)`,
  `back[:, t] = trans_wins`.
- Initialize `dp[:, 0] = -1e9` except `dp[0, 0] = cost[0, 0]`.
- Keep the existing backtracking loop (O(L+T), already cheap) EXACTLY as-is
  so path semantics (tie-breaking: trans wins on >=) are unchanged.
- The old range pruning (`t_last`, `T - L + 1` bounds) may be dropped — the
  -1e9 initialization preserves correctness because any path counted at
  (L-1, T-1) must have made exactly L-1 transitions. Verify via the
  equivalence test below.

## Changes to `scripts/nar_train_align.py`

In the training loop, replace the per-sample `log_prob[b, :L, :T].detach()
.cpu().numpy()` (8 GPU syncs/step) with ONE batched transfer before the loop:
`log_prob_np = log_prob.detach().cpu().numpy()` then slice per sample.

## test_v10c_smoke.py: new section (d)

- Equivalence: on 20 random continuous cost matrices (varied L in [3, 60],
  T in [L, 400], values ~ N(0,1) so exact ties have measure zero), durations
  from the new `maximum_path_np` must EXACTLY equal a reference copy of the
  old per-cell implementation (embed the old implementation in the test as
  `_maximum_path_reference`).
- Timing: on L=100, T=800, new implementation must be >= 10x faster than the
  reference (report the ratio).

## Acceptance

- py_compile both files; smoke test all sections PASS including (d).
- Only mas.py, nar_train_align.py, test_v10c_smoke.py touched. No training.
