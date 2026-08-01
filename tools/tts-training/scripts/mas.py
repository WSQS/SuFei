"""Monotonic Alignment Search (MAS) for FS2 duration learning.

Implements the alignment strategy from Glow-TTS (Kim et al., 2020):
- Two projection layers map encoder states and mel targets into a shared space
- MAS (dynamic programming) finds the optimal monotonic alignment at each step
- The resulting hard durations train the duration predictor
- At inference, the alignment module is discarded entirely

This module is training-only. The inference model is identical to the
standard FastSpeech 2 architecture.
"""
import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    n = L - 1
    k = np.arange(L, dtype=np.float64)  # [L]
    out = np.zeros((L, T), dtype=np.float32)
    for t in range(T):
        a = scaling * (t + 1)
        b = scaling * (T - t)
        betaln_ab = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        logC = math.lgamma(n + 1) - np.array([math.lgamma(kk + 1) for kk in k]) \
               - np.array([math.lgamma(n - kk + 1) for kk in k])
        betaln_term = np.array([math.lgamma(kk + a) + math.lgamma(n - kk + b) \
                                - math.lgamma(kk + a + n - kk + b) for kk in k])
        out[:, t] = (logC + betaln_term - betaln_ab).astype(np.float32)
    return out


def maximum_path_np(cost):
    """MAS dynamic programming (NumPy, per-sample, vectorized over phonemes).

    Finds the monotonic alignment path that maximizes total cost.
    Each phoneme gets at least 1 frame (enforced by the DP structure:
    the path must advance through all L rows).

    Column-sweep over t = 1..T-1; the phoneme dimension is vectorized.
    The -1e9 initialization makes any path counted at (L-1, T-1) must
    have made exactly L-1 transitions, so old range pruning is dropped.

    Args:
        cost: [L, T] cost matrix (log-likelihood, higher = better)
    Returns:
        path: [L, T] with 1 on the optimal path, 0 elsewhere
    """
    L, T = cost.shape
    if T < L:
        T_pad = L
        cost_padded = np.full((L, T_pad), cost.min() - 1.0, dtype=np.float32)
        cost_padded[:, :T] = cost
        cost = cost_padded
        T = T_pad

    cost = np.ascontiguousarray(cost, dtype=np.float64)

    dp = np.full(L, -1e9, dtype=np.float64)
    dp[0] = cost[0, 0]
    back = np.zeros((L, T), dtype=np.int32)  # 0=stay (same row), 1=advance (from row i-1)

    shifted = np.empty(L, dtype=np.float64)
    max_buf = np.empty(L, dtype=np.float64)
    for t in range(1, T):
        shifted[0] = -1e9
        shifted[1:] = dp[:-1]
        np.maximum(dp, shifted, out=max_buf)             # tie: trans wins (>=)
        back[:, t] = shifted >= dp
        dp = max_buf + cost[:, t]

    # Backtrack from (L-1, T-1) — identical to the reference implementation
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


def mas_durations(cost):
    """Get per-phoneme durations from MAS.

    Args:
        cost: [L, T] cost matrix
    Returns:
        durations: [L] int — number of mel frames per phoneme
    """
    L, T = cost.shape
    original_T = T
    if T < L:
        T_pad = L
        cost_padded = np.full((L, T_pad), cost.min() - 1.0, dtype=np.float32)
        cost_padded[:, :T] = cost
        cost = cost_padded
        T = T_pad

    path = maximum_path_np(cost)
    durations = path.sum(axis=1)

    # Verify
    assert durations.sum() == T, f"MAS path covers {durations.sum()} != {T} frames"
    assert (durations > 0).all(), "MAS produced zero-duration phoneme"

    return durations


class ForwardSumLoss(nn.Module):
    """CTC forward-sum loss over all monotonic paths (NeMo / FastPitch style)."""

    def __init__(self, blank_logprob=-1.0):
        super().__init__()
        self.blank_logprob = blank_logprob

    def forward(self, attn_logits, phone_lens, mel_lens, prior=None):
        """attn_logits: [B, L, T] RAW (pre-softmax) alignment scores.
        phone_lens, mel_lens: [B] ints. Returns scalar loss (batch mean).

        If prior ([B, L_max, T_max] log-prior tensor) is given, it is added to
        attn_logits before slicing/softmax (v10d: prior-in-softmax).
        """
        if prior is not None:
            attn_logits = attn_logits + prior
        B = attn_logits.size(0)
        losses = []
        for b in range(B):
            Lb = int(phone_lens[b])
            Tb = int(mel_lens[b])
            s = attn_logits[b, :Lb, :Tb].transpose(0, 1)  # [Tb, Lb]
            blank = self.blank_logprob * torch.ones(
                Tb, 1, device=s.device, dtype=s.dtype
            )
            s = torch.cat([blank, s], dim=1)  # [Tb, Lb+1]
            logprob = F.log_softmax(s, dim=-1)  # [Tb, Lb+1]
            logprob = logprob.reshape(Tb, 1, Lb + 1)
            targets = torch.arange(
                1, Lb + 1, device=s.device, dtype=torch.long
            ).unsqueeze(0)
            loss_b = F.ctc_loss(
                logprob, targets,
                input_lengths=(Tb,), target_lengths=(Lb,),
                blank=0, zero_infinity=True,
            )
            losses.append(loss_b)
        return torch.stack(losses).mean()


class AlignmentModule(nn.Module):
    """NeMo-style conv alignment encoder + MAS for training-time alignment.

    Two conv stacks project encoder states and mel targets into a shared D-dim
    space; scores are negative L2 distances. MAS (dynamic programming) finds
    the optimal monotonic alignment to extract durations.

    This module is added to FastSpeech2 for training only.
    At inference, it is discarded entirely.
    """

    def __init__(self, d_model=256, n_mels=80, d_align=128):
        super().__init__()
        self.key_proj = nn.Sequential(            # text branch: [B, L, d_model]
            nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(d_model * 2, d_align, kernel_size=1),
        )
        self.query_proj = nn.Sequential(          # mel branch: [B, T, n_mels]
            nn.Conv1d(n_mels, n_mels * 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(n_mels * 2, n_mels, kernel_size=1), nn.ReLU(),
            nn.Conv1d(n_mels, d_align, kernel_size=1),
        )
        self.d_model = d_model
        self.n_mels = n_mels
        self.d_align = d_align
        self.temperature = 0.0005

    def compute_scores(self, enc_out, mel_target):
        """Raw L2-distance scores [B, L, T] (no softmax, no masking).

        scores[b, l, t] = -temperature * ||keys[b, l] - queries[b, t]||^2
        Computed via the expanded form (||k||^2 + ||q||^2 - 2 k·q) to keep
        memory at [B, L, T] instead of [B, L, T, D].
        """
        # keys: [B, L, D]
        keys = self.key_proj(enc_out.transpose(1, 2)).transpose(1, 2)
        # queries: [B, T, D]
        queries = self.query_proj(mel_target.transpose(1, 2)).transpose(1, 2)

        k_sq = (keys ** 2).sum(-1, keepdim=True)          # [B, L, 1]
        q_sq = (queries ** 2).sum(-1, keepdim=True)        # [B, T, 1]
        kq = torch.bmm(keys, queries.transpose(1, 2))      # [B, L, T]
        dist_sq = k_sq + q_sq.transpose(1, 2) - 2.0 * kq   # [B, L, T]
        scores = -self.temperature * dist_sq
        return scores

    def compute_log_prob(self, enc_out, mel_target, enc_mask=None, mel_mask=None):
        """Compute log-probability cost matrix.

        Args:
            enc_out: [B, L, D_model] encoder hidden states
            mel_target: [B, T, n_mels] mel target (normalized)
            enc_mask: [B, L] True=padding (phonemes)
            mel_mask: [B, T] True=padding (mel frames)
        Returns:
            log_prob: [B, L, T] log P(phoneme i | frame t)
        """
        attn = self.compute_scores(enc_out, mel_target)  # [B, L, T]

        # Log-softmax over phoneme dimension for each frame
        log_prob = F.log_softmax(attn, dim=1)  # [B, L, T]

        if mel_mask is not None:
            # Mask padded frames
            log_prob = log_prob.masked_fill(mel_mask.unsqueeze(1), -1e4)

        return log_prob

    def get_durations(self, log_prob, phone_lens, mel_lens):
        """Run MAS per sample to extract durations.

        Args:
            log_prob: [B, L, T]
            phone_lens: [B] actual phoneme lengths
            mel_lens: [B] actual mel lengths
        Returns:
            durations: [B, L] int
            paths: [B, L, T] one-hot alignment paths
        """
        B, L_max, T_max = log_prob.shape
        durations_batch = torch.zeros(B, L_max, dtype=torch.long)
        paths_batch = torch.zeros(B, L_max, T_max, dtype=torch.float32)

        for b in range(B):
            L = int(phone_lens[b])
            T = int(mel_lens[b])
            cost = log_prob[b, :L, :T].detach().cpu().numpy().astype(np.float64)
            path = maximum_path_np(cost)  # [L, T]
            dur = path.sum(axis=1)  # [L]
            durations_batch[b, :L] = torch.from_numpy(dur).long()
            paths_batch[b, :L, :T] = torch.from_numpy(path).float()

        return durations_batch, paths_batch

    def alignment_loss(self, log_prob, paths, phone_mask, mel_mask):
        """Negative log-likelihood along the alignment path.

        Args:
            log_prob: [B, L, T]
            paths: [B, L, T] one-hot
            phone_mask: [B, L] True=padding
            mel_mask: [B, T] True=padding
        Returns:
            scalar loss
        """
        # Mask: only count valid (phoneme, frame) pairs
        valid = (~mel_mask).unsqueeze(1) & (~phone_mask).unsqueeze(2)  # [B, L, T]
        n_valid = valid.float().sum().clamp(min=1)

        # Loss = -sum(log_prob * path) / n_valid
        loss = -(log_prob * paths * valid.float()).sum() / n_valid

        return loss
