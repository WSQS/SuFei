"""v10c smoke test: prior + ForwardSumLoss + MAS-with-prior recovery.

CPU-only, no data files. Four sections:
  (a) log_beta_binomial_prior shape/finiteness/diagonal monotonicity
  (b) ForwardSumLoss finite + decreases under optimization
  (c) MAS + prior recovers a known block-diagonal alignment
  (d) vectorized maximum_path_np equivalence + speedup vs reference
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mas import (
    AlignmentModule,
    ForwardSumLoss,
    log_beta_binomial_prior,
    maximum_path_np,
    mas_durations,
)

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)
    return ok


# ─── Section (a): log_beta_binomial_prior ──────────────────────────────
def section_a():
    print("\n(a) log_beta_binomial_prior(20, 200)")
    prior = log_beta_binomial_prior(20, 200)
    check("shape == (20, 200)", prior.shape == (20, 200), str(prior.shape))
    check("all finite", np.all(np.isfinite(prior)))
    argmax_per_frame = prior.argmax(axis=0)
    diffs = np.diff(argmax_per_frame)
    nonneg = np.all(diffs >= 0)
    check(
        "per-frame argmax non-decreasing (roughly diagonal)",
        nonneg,
        f"min diff={diffs.min() if diffs.size else 0}",
    )


# ─── Section (b): ForwardSumLoss ───���───────────────────────────────────
def section_b():
    print("\n(b) AlignmentModule + ForwardSumLoss")
    torch.manual_seed(0)
    B, L_max, T_max = 2, 7, 55
    d_model, n_mels = 256, 80
    phone_lens = torch.tensor([5, 7])
    mel_lens = torch.tensor([40, 55])

    align = AlignmentModule(d_model=d_model, n_mels=n_mels, d_align=128)
    enc_out = torch.randn(B, L_max, d_model)
    mel_target = torch.randn(B, T_max, n_mels)
    scores = align.compute_scores(enc_out, mel_target)
    check("compute_scores shape [B, L, T]",
          scores.shape == (B, L_max, T_max), str(tuple(scores.shape)))
    check("compute_scores finite", torch.isfinite(scores).all().item())

    loss_fn = ForwardSumLoss()

    # --- prior=None ---
    loss = loss_fn(scores.detach(), phone_lens, mel_lens)
    check("loss (prior=None) is scalar", loss.dim() == 0, f"loss={loss.item():.4f}")
    check("loss (prior=None) finite", torch.isfinite(loss).item())

    # --- batched prior ---
    prior_t = torch.zeros(B, L_max, T_max)
    for b in range(B):
        Lb = int(phone_lens[b])
        Tb = int(mel_lens[b])
        prior_t[b, :Lb, :Tb] = torch.from_numpy(
            log_beta_binomial_prior(Lb, Tb, 1.0)
        )
    loss_p = loss_fn(scores.detach(), phone_lens, mel_lens, prior=prior_t)
    check("loss (prior) is scalar", loss_p.dim() == 0, f"loss={loss_p.item():.4f}")
    check("loss (prior) finite", torch.isfinite(loss_p).item())

    # --- decreases under optimization on the module's own params ---
    # Build synthetic block-diagonal alignment target and optimize the
    # AlignmentModule so its L2 scores align enc/mel pairs diagonally.
    L_eff, T_eff = 5, 40
    align_opt = AlignmentModule(d_model=d_model, n_mels=n_mels, d_align=128)
    opt = optim.Adam(align_opt.parameters(), lr=0.01)

    # Create paired (enc, mel) where mel is a length-regulated expansion of enc
    # so the true alignment is block-diagonal.
    gt_dur = np.array([8, 8, 8, 8, 8], dtype=np.int64)  # sum=40
    assert gt_dur.sum() == T_eff
    enc_synth = torch.randn(1, L_eff, d_model)
    mel_synth = torch.zeros(1, T_eff, n_mels)
    t_off = 0
    for i in range(L_eff):
        # make mel frames in this block a noisy copy of enc (broadcast to n_mels)
        d = int(gt_dur[i])
        mel_synth[0, t_off:t_off + d, :] = enc_synth[0, i, :n_mels].unsqueeze(0) \
            + 0.1 * torch.randn(d, n_mels)
        t_off += d

    prior_single = torch.from_numpy(
        log_beta_binomial_prior(L_eff, T_eff, 1.0)
    ).unsqueeze(0)  # [1, L, T]

    loss0 = None
    for step_i in range(60):
        opt.zero_grad()
        sc = align_opt.compute_scores(enc_synth, mel_synth)
        lo = loss_fn(sc, torch.tensor([L_eff]), torch.tensor([T_eff]),
                     prior=prior_single)
        if step_i == 0:
            loss0 = lo.item()
        lo.backward()
        opt.step()
    lossN = lo.item()
    check(
        "loss decreases after ~50 Adam steps on AlignmentModule params",
        lossN < loss0,
        f"{loss0:.4f} -> {lossN:.4f}",
    )


# ─── Section (c): MAS + prior recovers known alignment ─────────────────
def section_c():
    print("\n(c) MAS + prior recovers block-diagonal alignment")
    rng = np.random.default_rng(42)
    L, T = 10, 100
    gt_durations = np.array(
        [10, 8, 15, 5, 12, 7, 20, 6, 9, 8], dtype=np.float64
    )  # sum = 100
    assert gt_durations.sum() == T

    # Build a clean block-diagonal cost: high on the path, low elsewhere.
    cost = rng.normal(-2.0, 0.5, size=(L, T))
    t_offset = 0
    for i in range(L):
        d = int(gt_durations[i])
        cost[i, t_offset:t_offset + d] += 4.0
        t_offset += d

    # Add the prior (helps/guides DP) — should still recover well
    prior = log_beta_binomial_prior(L, T, 1.0).astype(np.float64)
    cost_with_prior = cost + prior

    dur_recovered = mas_durations(cost_with_prior)
    err = np.abs(dur_recovered - gt_durations)
    max_err = int(err.max())
    check(
        "max per-phoneme duration error <= 2",
        max_err <= 2,
        f"max_err={max_err}, err={err.tolist()}",
    )
    check(
        "recovered durations sum to T",
        int(dur_recovered.sum()) == T,
        f"sum={int(dur_recovered.sum())}",
    )


# ─── Section (d): vectorized DP equivalence + speedup ──────────────────
def _maximum_path_reference(cost):
    """Old per-cell reference implementation (embedded for equivalence test)."""
    L, T = cost.shape
    if T < L:
        T_pad = L
        cost_padded = np.full((L, T_pad), cost.min() - 1.0, dtype=np.float32)
        cost_padded[:, :T] = cost
        cost = cost_padded
        T = T_pad

    dp = np.full((L, T), -1e9, dtype=np.float64)
    back = np.zeros((L, T), dtype=np.int32)

    dp[0, 0] = cost[0, 0]
    for t in range(1, T - L + 1):
        dp[0, t] = dp[0, t - 1] + cost[0, t]

    for i in range(1, L):
        dp[i, i] = dp[i - 1, i - 1] + cost[i, i]
        back[i, i] = 1
        t_last = T - (L - 1 - i) - 1
        for t in range(i + 1, t_last + 1):
            stay = dp[i, t - 1]
            trans = dp[i - 1, t - 1]
            if trans >= stay:
                dp[i, t] = trans + cost[i, t]
                back[i, t] = 1
            else:
                dp[i, t] = stay + cost[i, t]
                back[i, t] = 0

    path = np.zeros((L, T), dtype=np.int32)
    t = T - 1
    for i in range(L - 1, -1, -1):
        while t >= 0:
            path[i, t] = 1
            if back[i, t] == 1:
                t -= 1
                break
            if t == 0:
                break
            t -= 1
    return path


def section_d():
    print("\n(d) vectorized maximum_path_np equivalence + speedup")
    rng = np.random.default_rng(123)

    # --- Equivalence over 20 random continuous cost matrices ---
    n_trials = 20
    max_mismatch = 0
    for trial in range(n_trials):
        L = int(rng.integers(3, 61))
        T = int(rng.integers(L, 401))
        cost = rng.normal(0.0, 1.0, size=(L, T)).astype(np.float64)
        # exact ties have measure zero with N(0,1) continuous values
        path_new = maximum_path_np(cost)
        path_ref = _maximum_path_reference(cost)
        mismatches = int((path_new != path_ref).sum())
        max_mismatch = max(max_mismatch, mismatches)

    check(
        f"paths EXACTLY match reference on {n_trials} random matrices",
        max_mismatch == 0,
        f"worst mismatch cells={max_mismatch}",
    )

    # --- Timing: L=100, T=800, new must be >= 10x faster ---
    L_bench, T_bench = 100, 800
    cost_bench = rng.normal(0.0, 1.0, size=(L_bench, T_bench)).astype(np.float64)

    # Warm up the new path to be fair
    _ = maximum_path_np(cost_bench)
    _ = _maximum_path_reference(cost_bench)

    n_reps = 3
    t0 = time.perf_counter()
    for _ in range(n_reps):
        _ = maximum_path_np(cost_bench)
    t_new = (time.perf_counter() - t0) / n_reps

    t0 = time.perf_counter()
    for _ in range(n_reps):
        _ = _maximum_path_reference(cost_bench)
    t_ref = (time.perf_counter() - t0) / n_reps

    ratio = t_ref / t_new if t_new > 0 else float("inf")
    check(
        f"new >= 10x faster than reference at L={L_bench},T={T_bench}",
        ratio >= 10.0,
        f"ref={t_ref * 1e3:.1f}ms new={t_new * 1e3:.1f}ms ratio={ratio:.1f}x",
    )
    print(f"        speedup ratio = {ratio:.1f}x "
          f"(ref {t_ref * 1e3:.1f}ms -> new {t_new * 1e3:.1f}ms)")


if __name__ == "__main__":
    print("=" * 60)
    print("v10c smoke test")
    print("=" * 60)
    section_a()
    section_b()
    section_c()
    section_d()
    print("\n" + "=" * 60)
    if failures:
        print(f"RESULT: FAIL — {len(failures)} section(s) failed: {failures}")
        sys.exit(1)
    else:
        print("RESULT: PASS — all sections passed")
        sys.exit(0)
