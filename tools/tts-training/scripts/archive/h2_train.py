"""H2: Mixed-window training for FastSpeech2.

Window distribution:
  50% full poem
  25% random single body line
  25% random adjacent body-line pair (couplet)

Everything else identical to H1:
  same train_171, same model (7.6M FS2), same LR/mask/loss/seed.
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
from torch.utils.data import Dataset, DataLoader

import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nar_fastspeech2 import FastSpeech2
from nar_train import masked_l1_loss, length_regulate_batch
from generate_paddlespeech_distillation_data import text_to_phonemes, PUNCT_TO_PHONE
from h2_linebound import compute_body_lines, DELIM_CHARS

CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"


# ─── Dataset ───────────────────────────────────────────────────────────

class MixedWindowDataset(Dataset):
    """Dataset that returns full poem / single line / couplet windows.

    Window type is sampled dynamically per __getitem__ call:
      50% full, 25% single line, 25% couplet.
    """

    def __init__(self, manifest_path, feature_dir, max_mel_len=2000,
                 p_full=0.50, p_line=0.25, p_couplet=0.25, seed=42):
        self.records = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["mel_len"] <= max_mel_len:
                    r["_npz_path"] = str(feature_dir / f"{r['poem_id']}.npz")
                    self.records.append(r)

        self.p_full = p_full
        self.p_line = p_line
        self.p_couplet = p_couplet
        self._rng = random.Random(seed)

        # Pre-compute body line segments for every poem
        self.line_segments = {}
        for r in self.records:
            segs = compute_body_lines(r["text"], r["phoneme_ids"], r["durations"])
            self.line_segments[r["poem_id"]] = segs

        n_with_lines = sum(1 for v in self.line_segments.values() if len(v) > 0)
        total_segs = sum(len(v) for v in self.line_segments.values())
        print(f"MixedWindowDataset: {len(self.records)} poems, "
              f"{n_with_lines} with body lines, "
              f"{total_segs} total body segments")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        roll = self._rng.random()
        segs = self.line_segments.get(r["poem_id"], [])

        if roll < self.p_full or len(segs) == 0:
            return self._get_full(r)
        elif roll < self.p_full + self.p_line or len(segs) == 1:
            seg = self._rng.choice(segs)
            return self._get_window(r, seg)
        else:
            # couplet: adjacent pair
            if len(segs) < 2:
                seg = self._rng.choice(segs)
                return self._get_window(r, seg)
            si = self._rng.randint(0, len(segs) - 2)
            return self._get_window_multi(r, [segs[si], segs[si + 1]])

    def _get_full(self, r):
        npz = np.load(r["_npz_path"])
        return {
            "poem_id": r["poem_id"],
            "window": "full",
            "phoneme_ids": np.array(r["phoneme_ids"], dtype=np.int64),
            "durations": np.array(r["durations"], dtype=np.int32),
            "mel": npz["mel"].astype(np.float32),
            "f0": npz["f0"].astype(np.float32),
            "energy": npz["energy"].astype(np.float32),
            "mel_len": r["mel_len"],
            "n_phonemes": len(r["phoneme_ids"]),
        }

    def _get_window(self, r, seg):
        npz = np.load(r["_npz_path"])
        ps, pe = seg["phone_start"], seg["phone_end"]
        fs, fe = seg["frame_start"], seg["frame_end"]
        return {
            "poem_id": r["poem_id"],
            "window": "line",
            "phoneme_ids": np.array(r["phoneme_ids"][ps:pe], dtype=np.int64),
            "durations": np.array(r["durations"][ps:pe], dtype=np.int32),
            "mel": npz["mel"][fs:fe].astype(np.float32),
            "f0": npz["f0"][fs:fe].astype(np.float32),
            "energy": npz["energy"][fs:fe].astype(np.float32),
            "mel_len": fe - fs,
            "n_phonemes": pe - ps,
        }

    def _get_window_multi(self, r, segs):
        """Concatenate adjacent segments (no cross-poem, contiguous in original)."""
        ps = segs[0]["phone_start"]
        pe = segs[-1]["phone_end"]
        fs = segs[0]["frame_start"]
        fe = segs[-1]["frame_end"]
        npz = np.load(r["_npz_path"])
        return {
            "poem_id": r["poem_id"],
            "window": "couplet",
            "phoneme_ids": np.array(r["phoneme_ids"][ps:pe], dtype=np.int64),
            "durations": np.array(r["durations"][ps:pe], dtype=np.int32),
            "mel": npz["mel"][fs:fe].astype(np.float32),
            "f0": npz["f0"][fs:fe].astype(np.float32),
            "energy": npz["energy"][fs:fe].astype(np.float32),
            "mel_len": fe - fs,
            "n_phonemes": pe - ps,
        }


def collate_fn(batch):
    """Pad batch to uniform shapes (same as nar_train.collate_fn)."""
    B = len(batch)
    phone_lens = [len(b["phoneme_ids"]) for b in batch]
    max_phone_len = max(phone_lens)
    mel_lens = [b["mel_len"] for b in batch]
    max_mel_len = max(mel_lens)

    phoneme_ids = np.zeros((B, max_phone_len), dtype=np.int64)
    durations = np.zeros((B, max_phone_len), dtype=np.int32)
    mel = np.zeros((B, max_mel_len, 80), dtype=np.float32)
    f0 = np.zeros((B, max_mel_len), dtype=np.float32)
    energy = np.zeros((B, max_mel_len), dtype=np.float32)

    phone_mask = np.ones((B, max_phone_len), dtype=bool)
    mel_mask = np.ones((B, max_mel_len), dtype=bool)

    for i, b in enumerate(batch):
        pl = phone_lens[i]
        ml = mel_lens[i]
        phoneme_ids[i, :pl] = b["phoneme_ids"]
        durations[i, :pl] = b["durations"]
        mel[i, :ml] = b["mel"]
        f0[i, :ml] = b["f0"]
        energy[i, :ml] = b["energy"]
        phone_mask[i, :pl] = False
        mel_mask[i, :ml] = False

    return {
        "poem_ids": [b["poem_id"] for b in batch],
        "windows": [b["window"] for b in batch],
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


# ─── Training ──────────────────────────────────────────────────────────

def train(args):
    device = args.device
    print(f"Device: {device}")

    if args.manifest_override:
        manifest = Path(args.manifest_override)
        if not manifest.is_absolute():
            manifest = ROOT / args.manifest_override
        feature_dir = ROOT / "data" / "paddle_distill_features"
        stats_path = (
            Path(args.stats_override) if args.stats_override
            else ROOT / "data" / "train_171_norm_stats.json"
        )
        if not stats_path.is_absolute() and args.stats_override:
            stats_path = ROOT / args.stats_override
        print(f"MANIFEST: {manifest}")
    else:
        manifest = ROOT / "data" / "paddle_distill_manifest.jsonl"
        feature_dir = ROOT / "data" / "paddle_distill_features"
        stats_path = ROOT / "data" / "paddle_distill_norm_stats.json"

    dataset = MixedWindowDataset(
        manifest, feature_dir, max_mel_len=args.max_mel_len,
        p_full=args.p_full, p_line=args.p_line, p_couplet=args.p_couplet,
        seed=args.seed,
    )

    # Load normalization stats
    if stats_path.exists():
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
        print(f"Loaded stats from {stats_path}")
    else:
        raise FileNotFoundError(f"Stats not found: {stats_path}")

    mel_mean = torch.tensor(stats["mel_mean"], device=device)
    mel_std = torch.tensor(stats["mel_std"], device=device)
    f0_mean, f0_std = stats["f0_mean"], stats["f0_std"]
    energy_mean, energy_std = stats["energy_mean"], stats["energy_std"]

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )

    # Mean log duration for bias init
    all_dur = []
    for i in range(len(dataset)):
        item = dataset._get_full(dataset.records[i])
        all_dur.extend(item["durations"].tolist())
    mean_dur = sum(all_dur) / len(all_dur)
    mean_log_dur = math.log(max(mean_dur, 1.0))
    print(f"Mean duration: {mean_dur:.1f} frames, log={mean_log_dur:.3f}")

    # Model — identical to H1
    model = FastSpeech2(
        vocab_size=268, d_model=256, nhead=2,
        num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=1024, n_mels=80, dropout=0.1,
        mean_log_dur=mean_log_dur,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params / 1e6:.1f}M)")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9
    )

    def lr_lambda(step):
        warmup = args.warmup_steps
        if step < warmup:
            return (step + 1) / warmup
        return max(0.1, 1.0 - (step - warmup) / max(1, args.steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(
        LOG_DIR / f"h2_train_{int(time.time())}.log", "w", encoding="utf-8"
    )

    W_MEL, W_DUR, W_PITCH, W_ENERGY = (
        args.w_mel, args.w_dur, args.w_pitch, args.w_energy
    )

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"H2_{int(time.time())}",
        config={
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "w_mel": W_MEL, "w_dur": W_DUR,
            "w_pitch": W_PITCH, "w_energy": W_ENERGY,
            "d_model": 256, "n_params": n_params,
            "n_samples": len(dataset),
            "mean_log_dur": mean_log_dur,
            "experiment": "H2_mixed_window",
            "p_full": args.p_full,
            "p_line": args.p_line,
            "p_couplet": args.p_couplet,
            "seed": args.seed,
        },
    )

    print(f"\n{'=' * 70}")
    print(f"H2 Mixed-Window Training started")
    print(f"  p_full={args.p_full} p_line={args.p_line} p_couplet={args.p_couplet}")
    print(f"{'=' * 70}")

    step = 0
    epoch = 0
    t_start = time.time()

    # Frame / exposure tracking
    cum_frames = 0
    win_counts = {"full": 0, "line": 0, "couplet": 0}

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

            # Track window types and effective frames
            for w in batch["windows"]:
                win_counts[w] += 1
            cum_frames += int(mel_lens.sum().item())

            # Normalize targets
            mel_norm = (mel_gt - mel_mean) / mel_std
            f0_norm = torch.where(
                f0_gt > 0,
                (torch.log(f0_gt.clamp(min=1)) - f0_mean) / f0_std,
                torch.zeros_like(f0_gt),
            )
            energy_norm = (energy_gt - energy_mean) / energy_std

            # Forward (teacher-forced: GT durations + pitch + energy)
            x = model.embedding(phoneme_ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x, mask=phone_mask)

            log_dur_pred = model.duration_predictor(x)
            pitch_pred_enc = model.pitch_predictor(x)
            energy_pred_enc = model.energy_predictor(x)

            mel_input = length_regulate_batch(x, durations_gt)
            T_pred = mel_input.size(1)
            T_gt = mel_gt.size(1)
            T_min = min(T_pred, T_gt)

            if args.gt_variance:
                T_var = min(T_pred, T_gt)
                pitch_expanded = f0_norm[:, :T_var]
                energy_expanded = energy_norm[:, :T_var]
                log_dur_pred = log_dur_pred.detach()
                pitch_pred_enc = pitch_pred_enc.detach()
                energy_pred_enc = energy_pred_enc.detach()
            else:
                pitch_expanded = length_regulate_batch(
                    pitch_pred_enc.unsqueeze(-1), durations_gt
                ).squeeze(-1)
                energy_expanded = length_regulate_batch(
                    energy_pred_enc.unsqueeze(-1), durations_gt
                ).squeeze(-1)

            pitch_embed = model.pitch_embed(
                pitch_expanded[:, :T_pred].unsqueeze(-1)
            )
            energy_embed = model.energy_embed(
                energy_expanded[:, :T_pred].unsqueeze(-1)
            )

            mel_input = mel_input + pitch_embed + energy_embed
            mel_input = model.pos_enc(mel_input)

            dec = mel_input
            dec_mask = mel_mask[:, :T_pred] if args.decoder_mask else None
            for layer in model.decoder_layers:
                dec = layer(dec, mask=dec_mask)
            mel_pred = model.mel_linear(dec)

            # Losses
            mel_loss = masked_l1_loss(
                mel_pred[:, :T_min], mel_norm[:, :T_min], mel_mask[:, :T_min]
            )
            dur_clipped = durations_gt.float().clamp(min=1, max=100)
            log_dur_gt = torch.log(dur_clipped)
            L_phone = min(log_dur_pred.size(1), log_dur_gt.size(1))
            dur_loss = masked_l1_loss(
                log_dur_pred[:, :L_phone], log_dur_gt[:, :L_phone],
                phone_mask[:, :L_phone],
            )

            if args.gt_variance:
                pitch_pred_exp = length_regulate_batch(
                    pitch_pred_enc.unsqueeze(-1), durations_gt
                ).squeeze(-1)
                energy_pred_exp = length_regulate_batch(
                    energy_pred_enc.unsqueeze(-1), durations_gt
                ).squeeze(-1)
                T_pitch = min(pitch_pred_exp.size(1), f0_norm.size(1))
                pitch_loss = masked_l1_loss(
                    pitch_pred_exp[:, :T_pitch], f0_norm[:, :T_pitch],
                    mel_mask[:, :T_pitch],
                )
                T_energy = min(energy_pred_exp.size(1), energy_norm.size(1))
                energy_loss = masked_l1_loss(
                    energy_pred_exp[:, :T_energy], energy_norm[:, :T_energy],
                    mel_mask[:, :T_energy],
                )
            else:
                T_pitch = min(pitch_expanded.size(1), f0_norm.size(1))
                pitch_loss = masked_l1_loss(
                    pitch_expanded[:, :T_pitch], f0_norm[:, :T_pitch],
                    mel_mask[:, :T_pitch],
                )
                T_energy = min(energy_expanded.size(1), energy_norm.size(1))
                energy_loss = masked_l1_loss(
                    energy_expanded[:, :T_energy], energy_norm[:, :T_energy],
                    mel_mask[:, :T_energy],
                )

            total_loss = (
                W_MEL * mel_loss + W_DUR * dur_loss
                + W_PITCH * pitch_loss + W_ENERGY * energy_loss
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.log_interval == 0 or step == 1:
                elapsed = time.time() - t_start
                speed = step / elapsed
                lr = scheduler.get_last_lr()[0]
                total_win = sum(win_counts.values())
                msg = (
                    f"  step {step:5d}/{args.steps} | epoch {epoch} | "
                    f"loss={total_loss.item():.4f} "
                    f"mel={mel_loss.item():.4f} "
                    f"dur={dur_loss.item():.4f} "
                    f"pitch={pitch_loss.item():.4f} "
                    f"energy={energy_loss.item():.4f} | "
                    f"lr={lr:.6f} | {speed:.1f} step/s | "
                    f"frames={cum_frames} "
                    f"full={win_counts['full']} "
                    f"line={win_counts['line']} "
                    f"couplet={win_counts['couplet']}"
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
                    "frames/cumulative": cum_frames,
                    "windows/full": win_counts["full"],
                    "windows/line": win_counts["line"],
                    "windows/couplet": win_counts["couplet"],
                    "windows/full_ratio": win_counts["full"] / max(total_win, 1),
                })

            if step % args.save_interval == 0:
                ckpt_path = CHECKPOINT_DIR / f"H2_step{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "cum_frames": cum_frames,
                    "win_counts": win_counts,
                    "stats": {
                        "mel_mean": stats["mel_mean"].tolist(),
                        "mel_std": stats["mel_std"].tolist(),
                        "f0_mean": stats["f0_mean"],
                        "f0_std": stats["f0_std"],
                        "energy_mean": stats["energy_mean"],
                        "energy_std": stats["energy_std"],
                    },
                }, str(ckpt_path))
                print(f"  >> Saved: {ckpt_path.name}  cum_frames={cum_frames}")

    # Final checkpoint
    ckpt_path = CHECKPOINT_DIR / f"H2_final.pt"
    torch.save({
        "model": model.state_dict(),
        "step": step,
        "epoch": epoch,
        "cum_frames": cum_frames,
        "win_counts": win_counts,
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
    print(f"Cumulative frames: {cum_frames}")
    print(f"Window counts: {win_counts}")
    log_file.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H2 mixed-window training")
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_mel_len", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=6000)
    parser.add_argument("--w_mel", type=float, default=1.0)
    parser.add_argument("--w_dur", type=float, default=1.0)
    parser.add_argument("--w_pitch", type=float, default=1.0)
    parser.add_argument("--w_energy", type=float, default=1.0)
    parser.add_argument("--wandb_project", default="sufei-tts")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--gt_variance", action="store_true")
    parser.add_argument("--decoder_mask", action="store_true")
    parser.add_argument("--manifest_override", default=None)
    parser.add_argument("--stats_override", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p_full", type=float, default=0.50)
    parser.add_argument("--p_line", type=float, default=0.25)
    parser.add_argument("--p_couplet", type=float, default=0.25)
    args = parser.parse_args()
    train(args)
