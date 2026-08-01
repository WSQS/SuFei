# Duration-Shortfall Root Fix Spec (task #7 → run m3_v2)

## Background

The duration predictor's total predicted frames systematically fall to
~0.79x of the alignment target, reproduced across ALL three label sources
(teacher dur 0.79 / MAS-biaobei 0.80 / MAS-CosyVoice 0.789). Root causes,
both in the training objective of `scripts/nar_train_align.py`:

1. `clamp(min=1, max=100)` on the duration target (line ~267) truncates
   punctuation-pause tokens (real pauses run 200-300 frames; the phoneme
   vocab has dedicated punctuation tokens "，" "。" "？" "！" and MAS
   assigns silence spans to them).
2. Log-domain L1's per-token optimum is the log-median (≈ geometric mean),
   which systematically underestimates the SUM of a right-skewed
   distribution.

The ×1.27 inference-time scale is diagnostic-only and must NOT be baked in.
This spec fixes the objective instead.

## Changes — ONLY file to modify: `scripts/nar_train_align.py`

### 1. CLI flag `--dur_clamp_max` (type float, default 100.0)

Replaces the hardcoded max in:

```python
mas_dur_clamped = mas_durations.float().clamp(min=1, max=100)
```

If `args.dur_clamp_max <= 0`: no upper clamp (`.clamp(min=1)` only).
Otherwise `.clamp(min=1, max=args.dur_clamp_max)`. Default 100.0 keeps
today's behavior bit-identical.

### 2. CLI flag `--w_dursum` (type float, default 0.0)

Duration-sum consistency loss, computed right after `dur_loss`:

```python
if args.w_dursum > 0:
    valid = (~phone_mask[:, :L_phone]).float()
    # clamp(max=6.0) ≈ 403 frames/phone caps early-training exp blowup
    pred_sum_b = (torch.exp(log_dur_pred[:, :L_phone].clamp(max=6.0)) * valid).sum(dim=1)
    mel_lens_t = mel_lens.to(device).float()
    dursum_loss = (torch.abs(pred_sum_b - mel_lens_t) / mel_lens_t).mean()
else:
    dursum_loss = torch.tensor(0.0, device=device)
```

Note `phone_mask` convention: True = padding (see `masked_l1_loss` in
`nar_train.py`), hence the `~`. `mel_lens` is a CPU tensor from the batch.

Add `+ args.w_dursum * dursum_loss` to the `total_loss` sum. It applies in
BOTH gate phases (duration supervision exists in both; MAS durations sum
exactly to T, so the target is consistent with dur_loss).

Rationale (add as a short comment): d(pred_sum)/d(log_dur_i) =
exp(log_dur_i) — the gradient concentrates on the LONGEST tokens
(punctuation pauses), exactly where the shortfall lives; the per-token
log-L1 keeps anchoring ordinary phones. This is what makes it different
from (and better than) a uniform inference-time ×1.27.

### 3. CLI flag `--run_name` (type str, default `"v10"`)

Checkpoint filenames currently hardcode the prefix and OVERWRITE across
runs:

- `CHECKPOINT_DIR / f"v10_step{step}.pt"` → `f"{args.run_name}_step{step}.pt"`
- `CHECKPOINT_DIR / "v10_final.pt"` → `f"{args.run_name}_final.pt"`

Default `"v10"` preserves today's filenames.

### 4. Logging

In the per-`log_interval` console line, after `dur=...` insert
`dursum={dursum_loss.item():.4f}`; also compute
`pred_ratio = (pred_sum_b / mel_lens_t).mean().item()` when
`args.w_dursum > 0` (else 0.0) and append `pred_ratio={pred_ratio:.3f}`
after `dur_ratio=...`. Add to the wandb dict: `"loss/dursum"` and
`"metrics/pred_sum_ratio"`.

(`dur_ratio` today is MAS-vs-teacher; `pred_ratio` is the new
predictor-vs-T signal we watch converge to 1.0.)

## Acceptance

- `C:/Users/wsqsy/.conda/envs/open-webui/python.exe -m py_compile scripts/nar_train_align.py` passes.
- With none of the three flags passed, numerical behavior is bit-identical
  to today (same losses, same checkpoint names; only cosmetic 0.0 fields in
  logs).
- Do not run training. Do not modify any other file.

---

## Addendum (post-m3_v2): dropout-free pred_ratio metric

m3_v2 finding: the logged `pred_ratio` read ~0.91 at the end of training,
but the deterministic eval measured 0.803. The gap is dropout noise in the
train-mode duration predictor plus Jensen inflation (E[exp(x+eps)] >
exp(E[x])). The monitoring metric must reflect the deterministic predictor,
or it overstates progress.

Change in `scripts/nar_train_align.py`, inside the
`if step % args.log_interval == 0 or step == 1:` block — REPLACE the
current `pred_ratio` computation (`pred_ratio = (pred_sum_b / mel_lens_t).mean().item()`)
with an eval-mode re-run of just the duration predictor:

```python
if args.w_dursum > 0:
    model.duration_predictor.eval()
    with torch.no_grad():
        log_dur_eval = model.duration_predictor(x.detach())
        valid_lp = (~phone_mask[:, :L_phone]).float()
        pred_sum_eval = (torch.exp(log_dur_eval[:, :L_phone].clamp(max=6.0))
                         * valid_lp).sum(dim=1)
        pred_ratio = (pred_sum_eval / mel_lens_t).mean().item()
    model.duration_predictor.train()
else:
    pred_ratio = 0.0
```

(`x`, `phone_mask`, `L_phone`, `mel_lens_t` are all in scope; `mel_lens_t`
is only defined when `args.w_dursum > 0`, which this branch guards.)

The dursum LOSS itself stays exactly as-is (train-mode) — only the logged
metric changes. Everything else untouched. Acceptance: py_compile passes;
behavior with `--w_dursum 0` unchanged.

---

## Addendum 2 (post-m3_v3): the dursum LOSS must be dropout-free too

m3_v3 (w_dursum 0.5) stalled at deterministic pred_ratio ~0.84 while the
train-mode dursum loss sat at ~0.05: the loss optimizes the DROPOUT-NOISED
sum, which Jensen-inflates by ~+0.10 (E[exp(x+eps)] > exp(E[x]); measured
0.91-vs-0.803 on m3_v2 and 0.95-vs-0.84 on m3_v3). So the noisy sum
reaches T while the deterministic sum equilibrates ~10% short. The loss
must act on the deterministic prediction.

In `scripts/nar_train_align.py`:

1. REPLACE the dursum loss block with a second, eval-mode forward of just
   the duration predictor WITH gradients (dropout off; input `x` NOT
   detached — same gradient paths as the train-mode dur_loss forward):

```python
if args.w_dursum > 0:
    # Deterministic (dropout-free) forward WITH grad: Jensen's inequality
    # makes E[exp(noisy)] > exp(clean); optimizing the noisy sum leaves
    # the deterministic sum ~10% short (m3_v2/v3 stalled at 0.80/0.84).
    model.duration_predictor.eval()
    log_dur_det = model.duration_predictor(x)
    model.duration_predictor.train()
    valid = (~phone_mask[:, :L_phone]).float()
    pred_sum_b = (torch.exp(log_dur_det[:, :L_phone].clamp(max=6.0))
                  * valid).sum(dim=1)
    mel_lens_t = mel_lens.to(device).float()
    dursum_loss = (torch.abs(pred_sum_b - mel_lens_t) / mel_lens_t).mean()
else:
    dursum_loss = torch.tensor(0.0, device=device)
```

2. SIMPLIFY the pred_ratio metric block back to reusing `pred_sum_b`
   (it is deterministic now — the Addendum-1 extra no_grad forward is
   redundant, remove it):

```python
if args.w_dursum > 0:
    pred_ratio = (pred_sum_b / mel_lens_t).mean().item()
else:
    pred_ratio = 0.0
```

Note the eval()/train() bracket only toggles dropout — the predictor uses
LayerNorm, unaffected by mode. Gradients flow normally through an
eval-mode module.

Acceptance: py_compile passes; `--w_dursum 0` path bit-identical; only
`scripts/nar_train_align.py` touched.

---

## Addendum 3 (post-m3_v4): linear-domain duration anchor `--w_dur_linear`

Motivation (poem_0027 doubled-王 diagnosis): dursum fixed the TOTAL
(pred_ratio 0.994) but its gradient ∝ exp(log_dur_i) dumps the recovered
mass onto pause tokens and long vowels ("。" 5.8→14.6 frames, uang2
29.8→37.6 vs v1), producing audible pause-insertion + re-articulation.
The log-L1 dur_loss cannot resist this: log(37.6/29.8)=0.23 costs the
same as a short phone going 4→5. A LINEAR-domain per-token anchor makes
+8 frames on a long token 8× more expensive than +1 on a short one, so
the sum correction spreads proportionally instead of concentrating.

In `scripts/nar_train_align.py`, all changes guarded so that
`--w_dur_linear 0` (the default) is bit-identical to current behavior:

1. New CLI flag next to `--w_dursum`:

```python
parser.add_argument("--w_dur_linear", type=float, default=0.0,
                    help="Weight for linear-domain per-token duration "
                         "anchor (smooth L1 in frames vs MAS target)")
```

2. CHANGE the deterministic-forward condition from `if args.w_dursum > 0:`
   to `if args.w_dursum > 0 or args.w_dur_linear > 0:` (the det forward is
   shared), and inside that block keep dursum_loss guarded by its own
   weight, adding the linear anchor:

```python
if args.w_dursum > 0 or args.w_dur_linear > 0:
    model.duration_predictor.eval()
    log_dur_det = model.duration_predictor(x)
    model.duration_predictor.train()
    valid = (~phone_mask[:, :L_phone]).float()
    pred_frames = torch.exp(log_dur_det[:, :L_phone].clamp(max=6.0))
    pred_sum_b = (pred_frames * valid).sum(dim=1)
    mel_lens_t = mel_lens.to(device).float()
    if args.w_dursum > 0:
        dursum_loss = (torch.abs(pred_sum_b - mel_lens_t) / mel_lens_t).mean()
    else:
        dursum_loss = torch.tensor(0.0, device=device)
    if args.w_dur_linear > 0:
        lin_diff = F.smooth_l1_loss(
            pred_frames, mas_dur_clamped[:, :L_phone].detach(),
            beta=2.0, reduction="none")
        dur_lin_loss = (lin_diff * valid).sum() / valid.sum().clamp(min=1)
    else:
        dur_lin_loss = torch.tensor(0.0, device=device)
else:
    dursum_loss = torch.tensor(0.0, device=device)
    dur_lin_loss = torch.tensor(0.0, device=device)
```

3. Total loss: add `+ args.w_dur_linear * dur_lin_loss` after the
   `args.w_dursum * dursum_loss` term.

4. Console log line: add `durlin={dur_lin_loss.item():.4f} ` right after
   the dursum field. Wandb dict: add `"loss/durlin": dur_lin_loss.item()`.

Notes: `mas_dur_clamped` is already in scope (dur_loss target). Use the
DETERMINISTIC forward for the anchor — same Jensen argument as dursum.
The pred_ratio metric block is unchanged (still keys off w_dursum alone).

Planned first use (m3_v5): `--w_dursum 0.5 --w_dur_linear 0.03`.
Gradient balance check: dursum pushes a 30-frame token with
~w_dursum*exp(ld)/T ≈ 0.5*30/1000 = 0.015; the anchor resists overshoot
with ~w_lin*exp(ld)/N_tok ≈ 0.03*30/60 = 0.015 — same order, while short
in-tolerance phones (|diff|<beta) feel only a quadratic nudge. Calibrate
from the logged unweighted durlin value in the first 500 steps.

Acceptance: py_compile passes; `--w_dur_linear` absent/0 path
bit-identical (existing m3_v4 recipe unaffected); only
`scripts/nar_train_align.py` touched.
