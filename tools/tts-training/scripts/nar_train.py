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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nar_fastspeech2 import FastSpeech2

MANIFEST = ROOT / "data" / "train_manifest.jsonl"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"


# ─── Dataset ───────────────────────────────────────────────────────────

class TTSDataset(Dataset):
    def __init__(self, manifest_path, feature_dir, max_mel_len=2000):
        self.records = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["mel_len"] <= max_mel_len:
                    r["_npz_path"] = str(feature_dir / f"{r['poem_id']}.npz")
                    self.records.append(r)
        print(f"Dataset: {len(self.records)} samples (max_mel_len={max_mel_len})")

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


# ─── Length Regulator (batch, with padding) ────────────────────────────

def length_regulate_batch(x, durations):
    """Expand encoder output by durations, padded to max output length.

    x: [B, L, D]
    durations: [B, L] (int, mel frames per phoneme)

    Returns: [B, T_max, D] where T_max = max(sum(durations))
    """
    B, L, D = x.shape
    total_durs = durations.sum(dim=1)
    T_max = int(total_durs.max().item())
    if T_max == 0:
        T_max = 1

    output = torch.zeros(B, T_max, D, device=x.device, dtype=x.dtype)

    for b in range(B):
        pos = 0
        for i in range(L):
            d = int(durations[b, i].item())
            if d <= 0:
                continue
            end = min(pos + d, T_max)
            output[b, pos:end] = x[b, i]
            pos = end
            if pos >= T_max:
                break

    return output


# ─── Training ──────────────────────────────────────────────────────────

def train(args):
    device = args.device
    print(f"Device: {device}")

    # Dataset
    feature_dir = ROOT / "data" / "nar_features"
    dataset = TTSDataset(MANIFEST, feature_dir, max_mel_len=args.max_mel_len)

    # Compute or load normalization stats
    stats_path = ROOT / "data" / "norm_stats.json"
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
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume} (step {start_step})")
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
    print(f"\n{'='*70}")
    print(f"Training started")
    print(f"{'='*70}")

    step = start_step
    epoch = start_step * args.batch_size // len(dataset)
    t_start = time.time()

    while step < args.steps:
        epoch += 1
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

            # Forward (teacher-forced: use GT durations + pitch + energy)
            x = model.embedding(phoneme_ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x, mask=phone_mask)

            log_dur_pred = model.duration_predictor(x)
            pitch_pred_enc = model.pitch_predictor(x)
            energy_pred_enc = model.energy_predictor(x)

            # Length regulate with GT durations
            mel_input = length_regulate_batch(x, durations_gt)
            T_pred = mel_input.size(1)
            T_gt = mel_gt.size(1)
            T_min = min(T_pred, T_gt)

            # Expand pitch/energy to mel length
            pitch_expanded = length_regulate_batch(
                pitch_pred_enc.unsqueeze(-1), durations_gt
            ).squeeze(-1)
            energy_expanded = length_regulate_batch(
                energy_pred_enc.unsqueeze(-1), durations_gt
            ).squeeze(-1)

            # Use GT pitch/energy for embedding injection (teacher forcing)
            pitch_embed = model.pitch_embed(f0_norm[:, :T_pred].unsqueeze(-1))
            energy_embed = model.energy_embed(energy_norm[:, :T_pred].unsqueeze(-1))

            mel_input = mel_input + pitch_embed + energy_embed
            mel_input = model.pos_enc(mel_input)

            dec = mel_input
            for layer in model.decoder_layers:
                dec = layer(dec)
            mel_pred = model.mel_linear(dec)

            # ── Compute losses ──

            # Mel loss (L1, masked)
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

            # Pitch loss (L1 on expanded encoder predictions vs GT, masked)
            T_pitch = min(pitch_expanded.size(1), f0_norm.size(1))
            pitch_loss = masked_l1_loss(
                pitch_expanded[:, :T_pitch],
                f0_norm[:, :T_pitch],
                mel_mask[:, :T_pitch],
            )

            # Energy loss (L1 on expanded encoder predictions vs GT, masked)
            T_energy = min(energy_expanded.size(1), energy_norm.size(1))
            energy_loss = masked_l1_loss(
                energy_expanded[:, :T_energy],
                energy_norm[:, :T_energy],
                mel_mask[:, :T_energy],
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)
