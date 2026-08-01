"""M2: Train FS2 with MAS-based duration learning (v10).

Joint training: encoder + alignment module + duration predictor + decoder.
MAS produces durations on-the-fly from (phoneme, mel) pairs — no external
duration labels needed.

Phased training:
  Phase 1 (0..align_warmup): teacher dur for mel/pitch/energy loss,
    alignment module trains in parallel on MAS loss
  Phase 2 (align_warmup..end): MAS durations for everything

Warm-starts from v9 checkpoint. The AlignmentModule is fresh-initialized.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nar_fastspeech2 import FastSpeech2
from nar_train import (
    TTSDataset, collate_fn, compute_stats,
    masked_l1_loss, length_regulate_batch, pool_to_phoneme,
)
from mas import AlignmentModule, maximum_path_np, ForwardSumLoss, log_beta_binomial_prior

CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"


def train(args):
    device = args.device
    print(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Seed: {args.seed}")

    # ─── Data ────────────────────���─────────────────────────────────────

    manifest = ROOT / args.manifest
    feature_dir = ROOT / args.feature_dir
    stats_path = ROOT / args.stats

    dataset = TTSDataset(manifest, feature_dir, max_mel_len=args.max_mel_len)
    print(f"Dataset: {len(dataset)} samples from {manifest}")

    with open(stats_path) as f:
        raw = json.load(f)
    stats = {
        "mel_mean": np.array(raw["mel_mean"], dtype=np.float32),
        "mel_std": np.array(raw["mel_std"], dtype=np.float32),
        "f0_mean": raw["f0_mean"],
        "f0_std": raw["f0_std"],
        "energy_mean": raw["energy_mean"],
        "energy_std": raw["energy_std"],
    }

    mel_mean = torch.tensor(stats["mel_mean"], device=device)
    mel_std = torch.tensor(stats["mel_std"], device=device)
    f0_mean = stats["f0_mean"]
    f0_std = stats["f0_std"]
    energy_mean = stats["energy_mean"]
    energy_std = stats["energy_std"]

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )

    # ─── Model ─────────────────────────────────────────────────────────

    model = FastSpeech2(
        vocab_size=268, d_model=256, nhead=2,
        num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=1024, n_mels=80,
        dropout=0.1, mean_log_dur=2.7,
    ).to(device)

    # Warm-start from v9
    if args.warm_start_ckpt:
        ckpt = torch.load(str(args.warm_start_ckpt), map_location=device, weights_only=False)
        sd = {k: v for k, v in ckpt["model"].items() if not k.startswith("film_gen")}
        model.load_state_dict(sd)
        print(f"Warm-started from {args.warm_start_ckpt} (step={ckpt.get('step')})")

    # Alignment module (fresh init)
    align_module = AlignmentModule(d_model=256, n_mels=80, d_align=args.d_align).to(device)
    fsum_loss_fn = ForwardSumLoss()

    # If M1 checkpoint exists, load projection weights
    if args.align_ckpt:
        align_ckpt = torch.load(str(args.align_ckpt), map_location=device, weights_only=False)
        align_module.load_state_dict(align_ckpt["align_module"])
        print(f"Loaded alignment module from {args.align_ckpt}")

    n_params = sum(p.numel() for p in model.parameters())
    n_align = sum(p.numel() for p in align_module.parameters())
    print(f"Model params: {n_params:,} ({n_params/1e6:.1f}M) + align {n_align:,} ({n_align/1e3:.1f}K)")

    # ─── Optimizer ────────────────────────────────────────────────────

    all_params = list(model.parameters()) + list(align_module.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    def lr_lambda(step):
        warmup = args.warmup_steps
        if step < warmup:
            return (step + 1) / warmup
        return max(0.1, 1.0 - (step - warmup) / max(1, args.steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_DIR / f"train_v10_{int(time.time())}.log", "w", encoding="utf-8")

    W_MEL = args.w_mel
    W_DUR = args.w_dur
    W_PITCH = args.w_pitch
    W_ENERGY = args.w_energy
    W_ALIGN = args.w_align

    print(f"\nLoss weights: mel={W_MEL}, dur={W_DUR}, pitch={W_PITCH}, "
          f"energy={W_ENERGY}, align={W_ALIGN}")
    print(f"Steps: {args.steps}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Align warmup: {args.align_warmup} (teacher dur before, MAS dur after)")

    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"v10_mas_{int(time.time())}",
            config={
                "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
                "warmup_steps": args.warmup_steps, "w_mel": W_MEL, "w_dur": W_DUR,
                "w_pitch": W_PITCH, "w_energy": W_ENERGY, "w_align": W_ALIGN,
                "d_align": args.d_align, "align_warmup": args.align_warmup,
                "n_params": n_params, "n_align_params": n_align,
                "warm_start": args.warm_start_ckpt is not None,
            },
        )
    except Exception as e:
        print(f"[warn] wandb.init failed ({e}); retrying offline")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"v10_mas_{int(time.time())}",
            config={
                "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
                "warmup_steps": args.warmup_steps, "w_mel": W_MEL, "w_dur": W_DUR,
                "w_pitch": W_PITCH, "w_energy": W_ENERGY, "w_align": W_ALIGN,
                "d_align": args.d_align, "align_warmup": args.align_warmup,
                "n_params": n_params, "n_align_params": n_align,
                "warm_start": args.warm_start_ckpt is not None,
            },
            mode="offline",
        )

    # ─── Training loop ────────────────────────────────────────────────

    print(f"\n{'='*70}\nTraining v10 (MAS)\n{'='*70}")

    step = 0
    epoch = 0
    t_start = time.time()
    ema_pct_le2 = 1.0
    gate_opened = False

    while step < args.steps:
        epoch += 1
        for batch in loader:
            if step >= args.steps:
                break

            model.train()
            align_module.train()
            optimizer.zero_grad()

            phoneme_ids = batch["phoneme_ids"].to(device)
            durations_gt = batch["durations"].to(device)  # teacher dur (phase 1 only)
            mel_gt = batch["mel"].to(device)
            f0_gt = batch["f0"].to(device)
            energy_gt = batch["energy"].to(device)
            phone_mask = batch["phone_mask"].to(device)
            mel_mask = batch["mel_mask"].to(device)
            phone_lens = batch["phone_lens"]
            mel_lens = batch["mel_lens"]

            B = phoneme_ids.size(0)

            # Normalize targets
            mel_norm = (mel_gt - mel_mean) / mel_std
            f0_norm = torch.where(
                f0_gt > 0,
                (torch.log(f0_gt.clamp(min=1)) - f0_mean) / f0_std,
                torch.zeros_like(f0_gt),
            )
            energy_norm = (energy_gt - energy_mean) / energy_std

            # ── Encoder forward ──
            x = model.embedding(phoneme_ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x, mask=phone_mask)

            # ── Compute MAS alignment ──
            # x is detached: alignment losses train ONLY the aligner convs.
            # The prior is added INSIDE the softmax (v10d) so it shapes both
            # the forward-sum loss and the DP cost consistently.
            scores = align_module.compute_scores(x.detach(), mel_norm)

            # Assemble padded per-batch log-prior tensor [B, L_max, T_max]
            if args.prior_off:
                prior_t = None
                log_prob = F.log_softmax(scores, dim=1)
            else:
                L_max = scores.size(1)
                T_max = scores.size(2)
                prior_t = torch.zeros(B, L_max, T_max, device=device, dtype=scores.dtype)
                for b in range(B):
                    Lb = int(phone_lens[b])
                    Tb = int(mel_lens[b])
                    prior_t[b, :Lb, :Tb] = torch.from_numpy(
                        log_beta_binomial_prior(Lb, Tb, args.prior_scale)
                    ).to(device)
                log_prob = F.log_softmax(scores + prior_t, dim=1)
            log_prob = log_prob.masked_fill(mel_mask.unsqueeze(1), -1e4)

            # Run MAS per sample (prior already in log_prob; no numpy prior add)
            mas_durations = torch.zeros(B, x.size(1), dtype=torch.long, device=device)
            mas_paths = torch.zeros(B, x.size(1), mel_norm.size(1), dtype=torch.float32, device=device)

            log_prob_np = log_prob.detach().cpu().numpy()
            for b in range(B):
                L = int(phone_lens[b])
                T = int(mel_lens[b])
                cost = log_prob_np[b, :L, :T].astype(np.float64)
                path = maximum_path_np(cost)
                dur = path.sum(axis=1)
                mas_durations[b, :L] = torch.from_numpy(dur).long().to(device)
                mas_paths[b, :L, :T] = torch.from_numpy(path).float().to(device)

            # Forward-sum loss (CTC over all monotonic paths); prior-in-softmax
            fsum = fsum_loss_fn(scores, phone_lens, mel_lens, prior=prior_t)

            # Alignment loss (trains enc_proj, mel_proj, encoder)
            align_loss = align_module.alignment_loss(
                log_prob, mas_paths, phone_mask, mel_mask
            )

            # ── Duration predictor ──
            log_dur_pred = model.duration_predictor(x)

            # Duration target: MAS durations
            if args.dur_clamp_max <= 0:
                mas_dur_clamped = mas_durations.float().clamp(min=1)
            else:
                mas_dur_clamped = mas_durations.float().clamp(min=1, max=args.dur_clamp_max)
            log_mas_dur = torch.log(mas_dur_clamped)
            L_phone = log_dur_pred.size(1)
            dur_loss = masked_l1_loss(
                log_dur_pred[:, :L_phone],
                log_mas_dur[:, :L_phone].detach(),
                phone_mask[:, :L_phone],
            )

            # Duration-sum consistency loss: d(pred_sum)/d(log_dur_i) =
            # exp(log_dur_i) — gradient concentrates on the LONGEST tokens
            # (punctuation pauses), exactly where the shortfall lives; the
            # per-token log-L1 keeps anchoring ordinary phones.
            if args.w_dursum > 0 or args.w_dur_linear > 0:
                # Deterministic (dropout-free) forward WITH grad: Jensen's inequality
                # makes E[exp(noisy)] > exp(clean); optimizing the noisy sum leaves
                # the deterministic sum ~10% short (m3_v2/v3 stalled at 0.80/0.84).
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

            # ���─ Choose duration source for mel path ──
            use_mas = (step >= args.align_warmup) and (ema_pct_le2 < 0.20)
            if use_mas and not gate_opened:
                print(
                    f"  >> Phase gate OPEN at step {step} "
                    f"(ema_pct_le2={ema_pct_le2:.3f}) — switching to MAS durations",
                    flush=True,
                )
                gate_opened = True

            # Cold-start: when gate is CLOSED, skip decoder/mel/pitch/energy
            # entirely (no usable teacher durations). Only align + fsum + dur.
            skip_expansion = args.cold_start and not use_mas

            if not skip_expansion:
                if use_mas:
                    dur_for_expand = mas_durations
                else:
                    dur_for_expand = durations_gt

                # ── Length regulate ──
                mel_input = length_regulate_batch(x, dur_for_expand)
                T_pred = mel_input.size(1)
                T_gt = mel_gt.size(1)
                T_min = min(T_pred, T_gt)

                # ── Pitch/energy predictors ──
                pitch_pred_enc = model.pitch_predictor(x)
                energy_pred_enc = model.energy_predictor(x)

                pitch_expanded = length_regulate_batch(
                    pitch_pred_enc.unsqueeze(-1), dur_for_expand
                ).squeeze(-1)
                energy_expanded = length_regulate_batch(
                    energy_pred_enc.unsqueeze(-1), dur_for_expand
                ).squeeze(-1)

                # ── Pitch/energy loss (frame-level) ──
                T_loss = min(pitch_expanded.size(1), f0_norm.size(1), T_pred)
                pitch_loss = masked_l1_loss(
                    pitch_expanded[:, :T_loss],
                    f0_norm[:, :T_loss],
                    mel_mask[:, :T_loss],
                )
                energy_loss = masked_l1_loss(
                    energy_expanded[:, :T_loss],
                    energy_norm[:, :T_loss],
                    mel_mask[:, :T_loss],
                )

                # ── Decode ──
                pitch_embed = model.pitch_embed(pitch_expanded[:, :T_pred].unsqueeze(-1))
                energy_embed = model.energy_embed(energy_expanded[:, :T_pred].unsqueeze(-1))
                mel_input = mel_input + pitch_embed + energy_embed
                mel_input = model.pos_enc(mel_input)

                dec = mel_input
                for layer in model.decoder_layers:
                    dec = layer(dec)
                mel_pred = model.mel_linear(dec)

                # ── Mel loss ──
                mel_loss = masked_l1_loss(
                    mel_pred[:, :T_min],
                    mel_norm[:, :T_min],
                    mel_mask[:, :T_min],
                )
            else:
                mel_loss = torch.tensor(0.0, device=device)
                pitch_loss = torch.tensor(0.0, device=device)
                energy_loss = torch.tensor(0.0, device=device)

            # ── Total loss ──
            total_loss = (
                W_MEL * mel_loss
                + W_DUR * dur_loss
                + W_PITCH * pitch_loss
                + W_ENERGY * energy_loss
                + W_ALIGN * align_loss
                + args.w_fsum * fsum
                + args.w_dursum * dursum_loss
                + args.w_dur_linear * dur_lin_loss
            )
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            # ── Update ema_pct_le2 every step (v10d phase gate signal) ──
            n_valid_phones = 0
            n_le2 = 0
            for b in range(B):
                L = int(phone_lens[b])
                md = mas_durations[b, :L]
                n_valid_phones += L
                n_le2 += int((md <= 2).sum().item())
            batch_pct_le2 = n_le2 / max(n_valid_phones, 1)
            ema_pct_le2 = 0.98 * ema_pct_le2 + 0.02 * batch_pct_le2

            if step % args.log_interval == 0 or step == 1:
                elapsed = time.time() - t_start
                lr = scheduler.get_last_lr()[0]
                phase = "MAS" if use_mas else "warmup"

                # Compute MAS-vs-teacher dur correlation for this batch
                dur_corrs = []
                dur_ratio = 0.0
                for b in range(B):
                    L = int(phone_lens[b])
                    md = mas_durations[b, :L].float()
                    td = durations_gt[b, :L].float()
                    if L > 1 and md.std() > 0 and td.std() > 0:
                        dur_corrs.append(torch.corrcoef(torch.stack([md, td]))[0, 1].item())
                    dur_ratio += md.sum().item() / max(td.sum().item(), 1)
                mean_dur_corr = np.mean(dur_corrs) if dur_corrs else 0.0
                mean_dur_ratio = dur_ratio / B

                if args.w_dursum > 0:
                    pred_ratio = (pred_sum_b / mel_lens_t).mean().item()
                else:
                    pred_ratio = 0.0

                msg = (
                    f"  step {step:5d}/{args.steps} | epoch {epoch} | [{phase}] "
                    f"loss={total_loss.item():.4f} "
                    f"mel={mel_loss.item():.4f} "
                    f"dur={dur_loss.item():.4f} "
                    f"dursum={dursum_loss.item():.4f} "
                    f"durlin={dur_lin_loss.item():.4f} "
                    f"pitch={pitch_loss.item():.4f} "
                    f"energy={energy_loss.item():.4f} "
                    f"align={align_loss.item():.4f} "
                    f"fsum={fsum.item():.4f} | "
                    f"dur_corr={mean_dur_corr:.3f} "
                    f"dur_ratio={mean_dur_ratio:.3f} "
                    f"pred_ratio={pred_ratio:.3f} "
                    f"pct_le2={batch_pct_le2:.3f} "
                    f"ema_le2={ema_pct_le2:.3f} | "
                    f"lr={lr:.6f} | {step/elapsed:.1f} step/s"
                )
                print(msg, flush=True)
                log_file.write(msg + "\n")
                log_file.flush()

                wandb.log({
                    "loss/total": total_loss.item(),
                    "loss/mel": mel_loss.item(),
                    "loss/dur": dur_loss.item(),
                    "loss/pitch": pitch_loss.item(),
                    "loss/energy": energy_loss.item(),
                    "loss/align": align_loss.item(),
                    "loss/fsum": fsum.item(),
                    "loss/dursum": dursum_loss.item(),
                    "loss/durlin": dur_lin_loss.item(),
                    "metrics/dur_corr": mean_dur_corr,
                    "metrics/dur_ratio": mean_dur_ratio,
                    "metrics/pct_dur_le2": batch_pct_le2,
                    "metrics/ema_pct_le2": ema_pct_le2,
                    "metrics/pred_sum_ratio": pred_ratio,
                    "lr": lr,
                    "step": step,
                    "epoch": epoch,
                    "phase": 1 if use_mas else 0,
                })

            if step % args.save_interval == 0:
                ckpt_path = CHECKPOINT_DIR / f"{args.run_name}_step{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "align_module": align_module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "stats": {
                        "mel_mean": stats["mel_mean"].tolist(),
                        "mel_std": stats["mel_std"].tolist(),
                        "f0_mean": stats["f0_mean"],
                        "f0_std": stats["f0_std"],
                        "energy_mean": stats["energy_mean"],
                        "energy_std": stats["energy_std"],
                    },
                }, str(ckpt_path))
                print(f"  >> Saved: {ckpt_path.name}", flush=True)

    # Final checkpoint
    ckpt_path = CHECKPOINT_DIR / f"{args.run_name}_final.pt"
    torch.save({
        "model": model.state_dict(),
        "align_module": align_module.state_dict(),
        "step": step,
        "epoch": epoch,
        "stats": {
            "mel_mean": stats["mel_mean"].tolist(),
            "mel_std": stats["mel_std"].tolist(),
            "f0_mean": stats["f0_mean"],
            "f0_std": stats["f0_std"],
            "energy_mean": stats["energy_mean"],
            "energy_std": stats["energy_std"],
        },
    }, str(ckpt_path))
    print(f"\nFinal checkpoint: {ckpt_path}")
    elapsed = time.time() - t_start
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log_file.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_mel_len", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=4000)
    parser.add_argument("--w_mel", type=float, default=1.0)
    parser.add_argument("--w_dur", type=float, default=1.0)
    parser.add_argument("--w_pitch", type=float, default=1.0)
    parser.add_argument("--w_energy", type=float, default=1.0)
    parser.add_argument("--w_align", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w_fsum", type=float, default=1.0)
    parser.add_argument("--prior_scale", type=float, default=1.0)
    parser.add_argument("--prior_off", action="store_true",
                        help="Disable the beta-binomial prior in MAS DP search")
    parser.add_argument("--d_align", type=int, default=128)
    parser.add_argument("--align_warmup", type=int, default=2000,
                        help="Use teacher dur for mel path during first N steps")
    parser.add_argument("--manifest", default="data/train_300_manifest_v3.jsonl")
    parser.add_argument("--feature_dir", default="data/paddle_distill_features_v3")
    parser.add_argument("--stats", default="data/train_300_norm_stats_v3.json")
    parser.add_argument("--warm_start_ckpt", default="checkpoints/fs2_final.pt",
                        help="v9 checkpoint to warm-start from")
    parser.add_argument("--align_ckpt", default=None,
                        help="M1 alignment checkpoint (optional)")
    parser.add_argument("--wandb_project", default="sufei-tts")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--cold_start", action="store_true",
                        help="Skip decoder/mel/pitch/energy while phase gate "
                             "is closed (no usable teacher durations)")
    parser.add_argument("--dur_clamp_max", type=float, default=100.0,
                        help="Max clamp for duration target (<=0 disables upper clamp)")
    parser.add_argument("--w_dursum", type=float, default=0.0,
                        help="Weight for duration-sum consistency loss")
    parser.add_argument("--w_dur_linear", type=float, default=0.0,
                        help="Weight for linear-domain per-token duration "
                             "anchor (smooth L1 in frames vs MAS target)")
    parser.add_argument("--run_name", type=str, default="v10",
                        help="Prefix for checkpoint filenames")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)
