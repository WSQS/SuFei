"""Stage 4: Batch training for FastSpeech 2 on all 219 teacher samples.

Improvements over overfit test:
- Proper batch collation with padding and masking
- Global mel/f0/energy normalization stats
- Masked losses (ignore padding positions)
- Duration loss with proper weighting
- Teacher-forced pitch/energy injection during training
- Checkpointing + logging
"""
import argparse
import json
import math
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

MANIFEST = ROOT / "data" / "train_manifest.jsonl"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"


# ─── Dataset ───────────────────────────────────────────────────────────

class TTSDataset(Dataset):
    def __init__(self, manifest_path, feature_dir, max_mel_len=2000, max_samples=0):
        self.records = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["mel_len"] <= max_mel_len:
                    r["_npz_path"] = str(feature_dir / f"{r['poem_id']}.npz")
                    self.records.append(r)
        if max_samples > 0:
            self.records.sort(key=lambda r: r["mel_len"])
            self.records = self.records[:max_samples]
        print(f"Dataset: {len(self.records)} samples (max_mel_len={max_mel_len}, max_samples={max_samples})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        npz = np.load(r["_npz_path"])
        return {
            "poem_id": r["poem_id"],
            "phoneme_ids": np.array(r["phoneme_ids"], dtype=np.int64),
            "durations": np.array(r["durations"], dtype=np.int32),
            "mel": npz["mel"].astype(np.float32),
            "f0": npz["f0"].astype(np.float32),
            "energy": npz["energy"].astype(np.float32),
            "mel_len": r["mel_len"],
            "n_phonemes": len(r["phoneme_ids"]),
        }


def collate_fn(batch):
    """Pad batch to uniform shapes."""
    B = len(batch)

    # Phoneme lengths
    phone_lens = [len(b["phoneme_ids"]) for b in batch]
    max_phone_len = max(phone_lens)

    # Mel lengths
    mel_lens = [b["mel_len"] for b in batch]
    max_mel_len = max(mel_lens)

    # Allocate padded arrays
    phoneme_ids = np.zeros((B, max_phone_len), dtype=np.int64)
    durations = np.zeros((B, max_phone_len), dtype=np.int32)
    mel = np.zeros((B, max_mel_len, 80), dtype=np.float32)
    f0 = np.zeros((B, max_mel_len), dtype=np.float32)
    energy = np.zeros((B, max_mel_len), dtype=np.float32)

    phone_mask = np.ones((B, max_phone_len), dtype=bool)  # True = padding
    mel_mask = np.ones((B, max_mel_len), dtype=bool)      # True = padding

    for i, b in enumerate(batch):
        pl = phone_lens[i]
        ml = mel_lens[i]
        phoneme_ids[i, :pl] = b["phoneme_ids"]
        durations[i, :pl] = b["durations"]
        mel[i, :ml] = b["mel"]
        f0[i, :ml] = b["f0"]
        energy[i, :ml] = b["energy"]
        phone_mask[i, pl:] = True
        phone_mask[i, :pl] = False
        mel_mask[i, ml:] = True
        mel_mask[i, :ml] = False

    return {
        "poem_ids": [b["poem_id"] for b in batch],
        "phoneme_ids": torch.from_numpy(phoneme_ids),
        "durations": torch.from_numpy(durations).long(),
        "mel": torch.from_numpy(mel),
        "f0": torch.from_numpy(f0),
        "energy": torch.from_numpy(energy),
        "phone_lens": torch.tensor(phone_lens),
        "mel_lens": torch.tensor(mel_lens),
        "phone_mask": torch.from_numpy(phone_mask),
        "mel_mask": torch.from_numpy(mel_mask),
    }


# ─── Normalization Stats ───────────────────────────────────────────────

def compute_stats(dataset):
    """Compute global mean/std for mel, f0, energy."""
    all_mel = []
    all_f0_voiced = []
    all_energy = []

    for i in range(len(dataset)):
        item = dataset[i]
        all_mel.append(item["mel"])
        f0 = item["f0"]
        voiced = f0[f0 > 0]
        if len(voiced) > 0:
            all_f0_voiced.append(np.log(voiced))
        all_energy.append(item["energy"])

    mel_cat = np.concatenate(all_mel, axis=0)
    mel_mean = mel_cat.mean(axis=0)
    mel_std = mel_cat.std(axis=0) + 1e-8

    f0_cat = np.concatenate(all_f0_voiced)
    f0_mean = f0_cat.mean()
    f0_std = f0_cat.std() + 1e-8

    energy_cat = np.concatenate(all_energy)
    energy_mean = energy_cat.mean()
    energy_std = energy_cat.std() + 1e-8

    return {
        "mel_mean": mel_mean,
        "mel_std": mel_std,
        "f0_mean": float(f0_mean),
        "f0_std": float(f0_std),
        "energy_mean": float(energy_mean),
        "energy_std": float(energy_std),
    }


# ─── Loss ──────────────────────────────────────────────────────────────

def masked_l1_loss(pred, target, mask):
    """L1 loss ignoring padded positions.

    pred: [B, T, D] or [B, T]
    target: same
    mask: [B, T] — True = padding (to ignore)
    """
    if mask.dim() == 2 and pred.dim() == 3:
        mask = mask.unsqueeze(-1)  # [B, T, 1]
    diff = (pred - target).abs() * (~mask).float()
    # Sum over all dims, normalize by number of valid elements
    n_valid = (~mask).float().sum().clamp(min=1)
    # But mask is [B, T, 1], so n_valid doesn't count D. 
    # Multiply by D for 3D case.
    if pred.dim() == 3:
        n_valid = n_valid * pred.size(-1)
    return diff.sum() / n_valid


def masked_mse_loss(pred, target, mask):
    """MSE loss ignoring padded positions."""
    if mask.dim() == 2 and pred.dim() == 3:
        mask = mask.unsqueeze(-1)
    diff = ((pred - target) ** 2) * (~mask).float()
    n_valid = (~mask).float().sum().clamp(min=1)
    if pred.dim() == 3:
        n_valid = n_valid * pred.size(-1)
    return diff.sum() / n_valid


def pool_to_phoneme(frame_values, durations, phone_mask):
    """Pool frame-level values to phoneme-level means (vectorized).

    frame_values: [B, T] (f0_norm or energy_norm)
    durations: [B, L] int (mel frames per phoneme)
    phone_mask: [B, L] bool (True=padding)

    Returns: [B, L] phoneme-level mean (0 for unvoiced/padding)
    """
    B, L = durations.shape
    T = frame_values.size(1)
    device = frame_values.device

    # Build frame-to-phoneme index via cumsum (same as length_regulate_batch)
    cumdurs = durations.cumsum(dim=1)  # [B, L]
    frame_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
    phoneme_per_frame = (
        (cumdurs.unsqueeze(2) > frame_idx.unsqueeze(1))  # [B, L, T]
        .float()
        .argmax(dim=1)  # [B, T]
    ).clamp(max=L - 1)

    # Mask out frames beyond total duration
    total_durs = durations.sum(dim=1, keepdim=True)  # [B, 1]
    valid_frame = (frame_idx < total_durs).float()  # [B, T]

    # Scatter-sum: for each phoneme, sum its frame values
    # Use one-hot expand: [B, T, L] → no, too much memory for large T/L
    # Instead use index_add_ per batch (still a Python loop but no .item())
    result = torch.zeros(B, L, device=device, dtype=frame_values.dtype)
    counts = torch.zeros(B, L, device=device, dtype=frame_values.dtype)
    for b in range(B):
        # segment_sum[b, l] = sum of frame_values[b, t] where phoneme_per_frame[b, t] == l
        result[b].index_add_(0, phoneme_per_frame[b], frame_values[b] * valid_frame[b])
        counts[b].index_add_(0, phoneme_per_frame[b], valid_frame[b])

    counts = counts.clamp(min=1)
    return result / counts


# ─── Length Regulator (batch, with padding) ────────────────────────────

def length_regulate_batch(x, durations):
    """Expand encoder output by durations, padded to max output length.

    Vectorized — no Python loops, no .item() syncs.
    x: [B, L, D]
    durations: [B, L] (int, mel frames per phoneme)
    Returns: [B, T_max, D]
    """
    B, L, D = x.shape
    durations = durations.clamp(min=0)
    total_durs = durations.sum(dim=1)
    T_max = int(total_durs.max().item())

    # Build segment ID per phoneme: [B, L] → each phoneme gets a unique segment index
    # Then expand to frame-level using cumsum trick
    # start_idx[b, l] = sum of durations[0..l-1]
    start_idx = durations.cumsum(dim=1) - durations  # [B, L]

    # Create frame-level index: [B, T_max]
    frame_idx = torch.arange(T_max, device=x.device).unsqueeze(0).expand(B, T_max)  # [B, T_max]

    # For each frame, find which phoneme it belongs to
    # phoneme_idx[b, t] = argmax of (start_idx <= t)
    # Use searchsorted equivalent: sum(start_idx <= t)
    # [B, T_max, L] comparison would be too much memory, so do it per-batch with gather

    output = torch.zeros(B, T_max, D, device=x.device, dtype=x.dtype)

    for b in range(B):
        d = durations[b]  # [L]
        starts = start_idx[b]  # [L]
        # For each frame t, find phoneme index = searchsorted(starts, t, right=True) - 1
        # But starts is sorted (cumsum), so we can use bucketize
        idx = torch.bucketize(frame_idx[b], starts + d, right=True)  # [T_max]
        idx = idx.clamp(max=L - 1)
        # Mask out frames beyond total duration
        mask = frame_idx[b] < total_durs[b]
        output[b] = x[b, idx] * mask.unsqueeze(-1)

    return output


# ─── Training ──────────────────────────────────────────────────────────

def train(args):
    device = args.device
    print(f"Device: {device}")

    # Dataset — use distillation manifest/features if --distill flag is set
    if args.manifest_override:
        manifest = Path(args.manifest_override)
        if not manifest.is_absolute():
            manifest = ROOT / args.manifest_override
        feature_dir = ROOT / "data" / "paddle_distill_features"
        stats_path = Path(args.stats_override) if args.stats_override else ROOT / "data" / "paddle_distill_norm_stats.json"
        if not stats_path.is_absolute():
            stats_path = ROOT / args.stats_override if args.stats_override else stats_path
        print(f"MANIFEST OVERRIDE: {manifest}")
    elif args.distill:
        manifest = ROOT / "data" / "paddle_distill_manifest.jsonl"
        feature_dir = ROOT / "data" / "paddle_distill_features"
        stats_path = ROOT / "data" / "paddle_distill_norm_stats.json"
        print("DISTILL MODE: using PaddleSpeech teacher data")
    else:
        manifest = MANIFEST
        feature_dir = ROOT / "data" / "nar_features"
        stats_path = ROOT / "data" / "norm_stats.json"

    dataset = TTSDataset(manifest, feature_dir, max_mel_len=args.max_mel_len, max_samples=args.max_samples)

    # Compute or load normalization stats
    if stats_path.exists() and not args.recompute_stats:
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
        print("Loaded normalization stats from cache")
    else:
        print("Computing normalization stats...")
        stats = compute_stats(dataset)
        with open(stats_path, "w") as f:
            json.dump({
                "mel_mean": stats["mel_mean"].tolist(),
                "mel_std": stats["mel_std"].tolist(),
                "f0_mean": stats["f0_mean"],
                "f0_std": stats["f0_std"],
                "energy_mean": stats["energy_mean"],
                "energy_std": stats["energy_std"],
            }, f, indent=2)
        print(f"  Saved to {stats_path}")

    print(f"  mel_mean range: [{stats['mel_mean'].min():.2f}, {stats['mel_mean'].max():.2f}]")
    print(f"  mel_std range:  [{stats['mel_std'].min():.4f}, {stats['mel_std'].max():.4f}]")
    print(f"  f0_mean={stats['f0_mean']:.2f}, f0_std={stats['f0_std']:.2f}")
    print(f"  energy_mean={stats['energy_mean']:.2f}, energy_std={stats['energy_std']:.2f}")

    # Move stats to device
    mel_mean = torch.tensor(stats["mel_mean"], device=device)
    mel_std = torch.tensor(stats["mel_std"], device=device)
    f0_mean = stats["f0_mean"]
    f0_std = stats["f0_std"]
    energy_mean = stats["energy_mean"]
    energy_std = stats["energy_std"]

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )

    # Validation set
    val_dataset = None
    if args.val_manifest:
        val_manifest = Path(args.val_manifest)
        if not val_manifest.is_absolute():
            val_manifest = ROOT / args.val_manifest
        val_feat_dir = feature_dir
        val_dataset = TTSDataset(val_manifest, val_feat_dir, max_mel_len=args.max_mel_len)
        print(f"Validation set: {len(val_dataset)} samples from {val_manifest}")

    # Compute mean log duration for bias init
    if not args.resume:
        all_dur = []
        for i in range(len(dataset)):
            all_dur.extend(dataset[i]["durations"].tolist())
        mean_dur = sum(all_dur) / len(all_dur)
        mean_log_dur = math.log(max(mean_dur, 1.0))
        print(f"Mean duration: {mean_dur:.1f} frames, log={mean_log_dur:.3f}")
    else:
        mean_log_dur = 2.7  # default, overridden by checkpoint

    # Model
    model = FastSpeech2(
        vocab_size=268,
        d_model=256,
        nhead=2,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        n_mels=80,
        dropout=0.1,
        mean_log_dur=mean_log_dur,
    ).to(device)

    start_step = 0
    if args.resume:
        ckpt = torch.load(str(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if not args.warm_start:
            start_step = ckpt.get("step", 0)
        else:
            # Reset predictors so they can learn from scratch
            if args.reset_predictors or True:  # Always reset for warm-start
                mean_log_dur_actual = mean_log_dur
                model.duration_predictor.init_bias(mean_log_dur_actual)
                model.pitch_predictor.init_bias(0.0)
                model.energy_predictor.init_bias(0.0)
                print(f"  Warm-start: reset all 3 variance predictors")
        print(f"Resumed from {args.resume} (warm_start={args.warm_start}, step counter={start_step})")
        if args.reset_dur_bias:
            model.duration_predictor.init_bias(mean_log_dur)
            print(f"  Reset duration predictor bias to {mean_log_dur:.3f}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # Warmup + linear decay
    def lr_lambda(step):
        warmup = args.warmup_steps
        if step < warmup:
            return (step + 1) / warmup
        return max(0.1, 1.0 - (step - warmup) / max(1, args.steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_DIR / f"train_{int(time.time())}.log", "w", encoding="utf-8")

    # Loss weights
    W_MEL = args.w_mel
    W_DUR = args.w_dur
    W_PITCH = args.w_pitch
    W_ENERGY = args.w_energy

    print(f"\nLoss weights: mel={W_MEL}, dur={W_DUR}, pitch={W_PITCH}, energy={W_ENERGY}")
    print(f"Steps: {args.steps}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Warmup: {args.warmup_steps} steps")

    # Wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"fs2_{int(time.time())}",
        config={
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "w_mel": W_MEL,
            "w_dur": W_DUR,
            "w_pitch": W_PITCH,
            "w_energy": W_ENERGY,
            "d_model": 256,
            "n_params": n_params,
            "n_samples": len(dataset),
            "mean_log_dur": mean_log_dur,
            "resumed": args.resume is not None,
            "distill": args.distill,
            "gt_variance": args.gt_variance,
            "full_e2e": args.full_e2e,
            "warm_start": args.warm_start,
            "reset_predictors": args.reset_predictors,
            "freeze_steps": args.freeze_steps,
            "full_e2e_steps": args.full_e2e_steps,
        },
    )

    print(f"\n{'='*70}")
    print(f"Training started")
    print(f"{'='*70}")

    @torch.no_grad()
    def run_validation(model, val_dataset, mel_mean_t, mel_std_t, f0_mean_v, f0_std_v,
                       energy_mean_v, energy_std_v):
        model.eval()
        l1s = []
        for i in range(len(val_dataset)):
            item = val_dataset[i]
            phone_ids = torch.tensor([item["phoneme_ids"]], dtype=torch.long, device=device)
            dur_gt = torch.tensor([item["durations"]], dtype=torch.long, device=device)
            mel_gt = torch.tensor(item["mel"]).unsqueeze(0).to(device)
            f0_gt = torch.tensor(item["f0"]).unsqueeze(0).to(device)
            e_gt = torch.tensor(item["energy"]).unsqueeze(0).to(device)
            mel_mask = torch.zeros(1, mel_gt.size(1), dtype=torch.bool, device=device)

            mel_norm = (mel_gt - mel_mean_t) / mel_std_t
            f0_norm = torch.where(
                f0_gt > 0,
                (torch.log(f0_gt.clamp(min=1)) - f0_mean_v) / f0_std_v,
                torch.zeros_like(f0_gt),
            )
            e_norm = (e_gt - energy_mean_v) / energy_std_v

            x = model.embedding(phone_ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x)

            if args.full_e2e or args.full_e2e_steps > 0:
                # Validate with predicted variance (matches training)
                log_dur = model.duration_predictor(x)
                pitch_p = model.pitch_predictor(x)
                energy_p = model.energy_predictor(x)
                pred_durations = log_dur.exp().round().clamp(min=1).long()
                mel_input = length_regulate_batch(x, pred_durations)
                T_out = mel_input.size(1)
                pitch_exp = length_regulate_batch(
                    pitch_p.unsqueeze(-1), pred_durations
                ).squeeze(-1)
                energy_exp = length_regulate_batch(
                    energy_p.unsqueeze(-1), pred_durations
                ).squeeze(-1)
                mel_input = mel_input + \
                    model.pitch_embed(pitch_exp[:, :T_out].unsqueeze(-1)) + \
                    model.energy_embed(energy_exp[:, :T_out].unsqueeze(-1))
            else:
                # Validate with GT variance (default)
                mel_input = length_regulate_batch(x, dur_gt)
                T_out = mel_input.size(1)
                mel_input = mel_input + \
                    model.pitch_embed(f0_norm[:, :T_out].unsqueeze(-1)) + \
                    model.energy_embed(e_norm[:, :T_out].unsqueeze(-1))
            mel_input = model.pos_enc(mel_input)
            dec = mel_input
            for layer in model.decoder_layers:
                dec = layer(dec)
            mel_pred = model.mel_linear(dec)

            T = min(mel_pred.size(1), mel_norm.size(1))
            l1 = (mel_pred[:, :T] - mel_norm[:, :T]).abs().mean().item()
            l1s.append(l1)

        model.train()
        return float(np.mean(l1s))

    step = start_step
    epoch = start_step * args.batch_size // len(dataset)
    t_start = time.time()

    # Freeze logic: train only predictors for first freeze_steps
    if args.freeze_steps > 0:
        freeze_params = set()
        for name, param in model.named_parameters():
            if not any(p in name for p in ["duration_predictor", "pitch_predictor", "energy_predictor"]):
                param.requires_grad = False
                freeze_params.add(name)
        print(f"FREEZE: {len(freeze_params)} param tensors frozen (encoder+decoder+embeds), training only predictors")

    while step < args.steps:
        epoch += 1

        # Unfreeze after freeze_steps
        if args.freeze_steps > 0 and step == args.freeze_steps:
            for name, param in model.named_parameters():
                param.requires_grad = True
            print(f"  >> UNFREEZE at step {step}: all parameters trainable")
        for batch in loader:
            if step >= args.steps:
                break

            model.train()
            optimizer.zero_grad()

            phoneme_ids = batch["phoneme_ids"].to(device)
            durations_gt = batch["durations"].to(device)
            mel_gt = batch["mel"].to(device)
            f0_gt = batch["f0"].to(device)
            energy_gt = batch["energy"].to(device)
            phone_mask = batch["phone_mask"].to(device)
            mel_mask = batch["mel_mask"].to(device)
            mel_lens = batch["mel_lens"].to(device)

            B = phoneme_ids.size(0)

            # Normalize targets
            mel_norm = (mel_gt - mel_mean) / mel_std
            f0_norm = torch.where(
                f0_gt > 0,
                (torch.log(f0_gt.clamp(min=1)) - f0_mean) / f0_std,
                torch.zeros_like(f0_gt),
            )
            energy_norm = (energy_gt - energy_mean) / energy_std

            # Forward
            x = model.embedding(phoneme_ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x, mask=phone_mask)

            log_dur_pred = model.duration_predictor(x)
            pitch_pred_enc = model.pitch_predictor(x)
            energy_pred_enc = model.energy_predictor(x)

            # Determine mode for this step (phased training)
            use_full_e2e = args.full_e2e
            if args.full_e2e_steps > 0:
                use_full_e2e = step >= args.full_e2e_steps

            if args.gt_variance:
                # B' mode: use GT durations + GT pitch/energy, detach predictors
                mel_input = length_regulate_batch(x, durations_gt)
                T_pred = mel_input.size(1)
                T_gt = mel_gt.size(1)
                T_min = min(T_pred, T_gt)

                T_var = min(T_pred, T_gt)
                pitch_expanded = f0_norm[:, :T_var]
                energy_expanded = energy_norm[:, :T_var]
                log_dur_pred = log_dur_pred.detach()
                pitch_pred_enc = pitch_pred_enc.detach()
                energy_pred_enc = energy_pred_enc.detach()
            elif use_full_e2e:
                # Full E2E: use PREDICTED durations + predicted pitch/energy
                # This eliminates the train/inference gap completely
                pred_durations = log_dur_pred.detach().exp().round().clamp(min=1).long()
                mel_input = length_regulate_batch(x, pred_durations)
                T_pred = mel_input.size(1)
                T_gt = mel_gt.size(1)
                T_min = min(T_pred, T_gt)

                pitch_expanded = length_regulate_batch(
                    pitch_pred_enc.unsqueeze(-1), pred_durations
                ).squeeze(-1)
                energy_expanded = length_regulate_batch(
                    energy_pred_enc.unsqueeze(-1), pred_durations
                ).squeeze(-1)
            else:
                # Default E2E: GT durations + predicted pitch/energy
                mel_input = length_regulate_batch(x, durations_gt)
                T_pred = mel_input.size(1)
                T_gt = mel_gt.size(1)
                T_min = min(T_pred, T_gt)

                pitch_expanded = length_regulate_batch(
                    pitch_pred_enc.unsqueeze(-1), durations_gt
                ).squeeze(-1)
                energy_expanded = length_regulate_batch(
                    energy_pred_enc.unsqueeze(-1), durations_gt
                ).squeeze(-1)

            pitch_embed = model.pitch_embed(pitch_expanded[:, :T_pred].unsqueeze(-1))
            energy_embed = model.energy_embed(energy_expanded[:, :T_pred].unsqueeze(-1))

            mel_input = mel_input + pitch_embed + energy_embed
            mel_input = model.pos_enc(mel_input)

            dec = mel_input
            dec_mask = mel_mask[:, :T_pred] if (args.decoder_mask and T_pred <= mel_mask.size(1)) else None
            for layer in model.decoder_layers:
                dec = layer(dec, mask=dec_mask)
            mel_pred = model.mel_linear(dec)

            # ── Compute losses ──

            # Mel loss (L1, masked) — compare at the shorter length
            mel_loss = masked_l1_loss(
                mel_pred[:, :T_min],
                mel_norm[:, :T_min],
                mel_mask[:, :T_min],
            )

            # Duration loss (L1 on log, masked on phone positions)
            # Clip extreme durations (punctuation pauses) to avoid MSE blowup
            dur_clipped = durations_gt.float().clamp(min=1, max=100)
            log_dur_gt = torch.log(dur_clipped)
            L_phone = min(log_dur_pred.size(1), log_dur_gt.size(1))
            dur_loss = masked_l1_loss(
                log_dur_pred[:, :L_phone],
                log_dur_gt[:, :L_phone],
                phone_mask[:, :L_phone],
            )

            # Pitch/energy loss — phoneme-level L1 (no frame-level floor)
            # Predictor outputs 1 scalar per phoneme, so compare against
            # phoneme-level mean of GT, not per-frame GT.
            # This eliminates the irreducible floor from within-phoneme contour.
            f0_phoneme_gt = pool_to_phoneme(f0_norm, durations_gt, phone_mask)
            energy_phoneme_gt = pool_to_phoneme(energy_norm, durations_gt, phone_mask)

            L_p = min(pitch_pred_enc.size(1), f0_phoneme_gt.size(1))
            pitch_loss = masked_l1_loss(
                pitch_pred_enc[:, :L_p],
                f0_phoneme_gt[:, :L_p],
                phone_mask[:, :L_p],
            )
            L_e = min(energy_pred_enc.size(1), energy_phoneme_gt.size(1))
            energy_loss = masked_l1_loss(
                energy_pred_enc[:, :L_e],
                energy_phoneme_gt[:, :L_e],
                phone_mask[:, :L_e],
            )

            total_loss = W_MEL * mel_loss + W_DUR * dur_loss + W_PITCH * pitch_loss + W_ENERGY * energy_loss
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.log_interval == 0 or step == 1:
                elapsed = time.time() - t_start
                speed = step / elapsed
                lr = scheduler.get_last_lr()[0]
                msg = (
                    f"  step {step:5d}/{args.steps} | epoch {epoch} | "
                    f"loss={total_loss.item():.4f} "
                    f"mel={mel_loss.item():.4f} "
                    f"dur={dur_loss.item():.4f} "
                    f"pitch={pitch_loss.item():.4f} "
                    f"energy={energy_loss.item():.4f} | "
                    f"lr={lr:.6f} | "
                    f"{speed:.1f} step/s"
                )
                print(msg)
                log_file.write(msg + "\n")
                log_file.flush()

                wandb.log({
                    "loss/total": total_loss.item(),
                    "loss/mel": mel_loss.item(),
                    "loss/dur": dur_loss.item(),
                    "loss/pitch": pitch_loss.item(),
                    "loss/energy": energy_loss.item(),
                    "lr": lr,
                    "step": step,
                    "epoch": epoch,
                })

            if val_dataset is not None and step % args.val_interval == 0:
                val_l1 = run_validation(
                    model, val_dataset,
                    mel_mean, mel_std,
                    f0_mean, f0_std,
                    energy_mean, energy_std,
                )
                val_msg = f"  [VAL] step {step:5d} | val_mel_l1={val_l1:.4f} | train_mel_l1={mel_loss.item():.4f}"
                print(val_msg)
                log_file.write(val_msg + "\n")
                log_file.flush()
                wandb.log({"val/mel_l1": val_l1, "step": step})

            if step % args.save_interval == 0:
                ckpt_path = CHECKPOINT_DIR / f"fs2_step{step}.pt"
                torch.save({
                    "model": model.state_dict(),
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
                print(f"  >> Saved checkpoint: {ckpt_path.name}")

    # Final checkpoint
    ckpt_path = CHECKPOINT_DIR / f"fs2_final.pt"
    torch.save({
        "model": model.state_dict(),
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
    print(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log_file.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_mel_len", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--recompute_stats", action="store_true")
    parser.add_argument("--w_mel", type=float, default=1.0)
    parser.add_argument("--w_dur", type=float, default=1.0)
    parser.add_argument("--w_pitch", type=float, default=1.0)
    parser.add_argument("--w_energy", type=float, default=1.0)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--reset_dur_bias", action="store_true", help="Re-init duration predictor bias on resume")
    parser.add_argument("--warm_start", action="store_true",
                        help="Warm-start: resume weights but reset step counter and predictors")
    parser.add_argument("--reset_predictors", action="store_true",
                        help="Re-init duration/pitch/energy predictors (for warm-start from gt-var model)")
    parser.add_argument("--freeze_steps", type=int, default=0,
                        help="Freeze encoder+decoder for first N steps, train only predictors")
    parser.add_argument("--full_e2e_steps", type=int, default=0,
                        help="Switch to full_e2e mode after N steps (before that, use GT dur + pred var)")
    parser.add_argument("--wandb_project", default="sufei-tts", help="WandB project name")
    parser.add_argument("--wandb_name", default=None, help="WandB run name")
    parser.add_argument("--distill", action="store_true", help="Use PaddleSpeech distillation data")
    parser.add_argument("--gt_variance", action="store_true", help="Use GT pitch/energy for decoder, detach predictors")
    parser.add_argument("--full_e2e", action="store_true", help="Full E2E: use predicted durations+pitch+energy for decoder (no teacher forcing)")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit dataset to first N samples (sorted by mel_len). 0 = no limit")
    parser.add_argument("--decoder_mask", action="store_true", help="Apply mel_mask to decoder self-attention")
    parser.add_argument("--manifest_override", default=None, help="Override manifest path (relative to ROOT or absolute)")
    parser.add_argument("--stats_override", default=None, help="Override norm stats path (relative to ROOT or absolute)")
    parser.add_argument("--val_manifest", default=None, help="Validation manifest for periodic mel L1 eval")
    parser.add_argument("--val_interval", type=int, default=100, help="Run validation every N steps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)
