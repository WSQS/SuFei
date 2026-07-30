"""Step 3a: Duration predictor D-warm vs D-reset comparison.

Freezes encoder + decoder + pitch/energy predictors from acoustic_baseline_step12000.
Only trains duration predictor. Compares:
  D-warm: keep existing duration predictor weights from step12000
  D-reset: reinitialize duration predictor (tiny weight + bias=log(mean_dur))

Each runs 500 steps on all 204 samples. Reports duration ratio, MAE, 0-frame ratio.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2
from nar_train import TTSDataset, collate_fn, masked_l1_loss

BASELINE = ROOT / "checkpoints" / "acoustic_baseline_step12000.pt"
MANIFEST = ROOT / "data" / "train_manifest.jsonl"
FEAT_DIR = ROOT / "data" / "nar_features"
OUT = ROOT / "checkpoints"

device = "cuda" if torch.cuda.is_available() else "cpu"


def freeze_acoustic(model):
    """Freeze everything except duration predictor."""
    for name, param in model.named_parameters():
        if "duration_predictor" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable: {n_trainable:,} (duration predictor)")
    print(f"  Frozen: {n_frozen:,} (encoder + decoder + pitch/energy)")


def train_dur_only(model, loader, steps, lr, label):
    """Train only duration predictor for given steps."""
    dur_params = [p for p in model.duration_predictor.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(dur_params, lr=lr, betas=(0.9, 0.98), eps=1e-9)

    step = 0
    t_start = time.time()

    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            model.train()
            optimizer.zero_grad()

            phoneme_ids = batch["phoneme_ids"].to(device)
            durations_gt = batch["durations"].to(device)
            phone_mask = batch["phone_mask"].to(device)

            # Forward through frozen encoder
            with torch.no_grad():
                x = model.embedding(phoneme_ids) * math.sqrt(model.d_model)
                x = model.pos_enc(x)
                for layer in model.encoder_layers:
                    x = layer(x, mask=phone_mask)

            # Duration predictor (trainable)
            log_dur_pred = model.duration_predictor(x)

            # Duration loss only
            dur_clipped = durations_gt.float().clamp(min=1, max=100)
            log_dur_gt = torch.log(dur_clipped)
            L = min(log_dur_pred.size(1), log_dur_gt.size(1))
            dur_loss = masked_l1_loss(
                log_dur_pred[:, :L], log_dur_gt[:, :L], phone_mask[:, :L]
            )
            dur_loss.backward()
            optimizer.step()
            step += 1

            if step % 100 == 0 or step == 1:
                pred_sum = log_dur_pred.detach().exp().sum().item()
                gt_sum = durations_gt.float().sum().item()
                ratio = pred_sum / max(gt_sum, 1)
                elapsed = time.time() - t_start
                print(f"  [{label}] step {step:4d}/{steps}: dur_loss={dur_loss.item():.4f} ratio={ratio:.1%} ({elapsed:.0f}s)")


def eval_duration(model, dataset):
    """Evaluate duration predictor on all samples."""
    model.eval()
    ratios = []
    all_pred = []
    all_gt = []

    for i in range(len(dataset)):
        item = dataset[i]
        ids = torch.tensor([item["phoneme_ids"]], device=device)
        gt_dur = np.array(item["durations"])

        with torch.no_grad():
            x = model.embedding(ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x)
            log_dur = model.duration_predictor(x)

        pred_dur = log_dur[0].exp().cpu().numpy()
        ratio = pred_dur.sum() / max(gt_dur.sum(), 1)
        ratios.append(ratio)
        all_pred.extend(pred_dur.tolist())
        all_gt.extend(gt_dur.tolist())

    ratios = np.array(ratios)
    all_pred = np.array(all_pred)
    all_gt = np.array(all_gt)

    # Per-frame MAE
    mae = np.abs(all_pred - all_gt).mean()

    # 0-frame and 1-frame ratio
    zero_frames = (all_pred < 0.5).sum() / len(all_pred)
    one_frames = ((all_pred >= 0.5) & (all_pred < 1.5)).sum() / len(all_pred)

    return {
        "median_ratio": float(np.median(ratios)),
        "mean_ratio": float(ratios.mean()),
        "p10_ratio": float(np.percentile(ratios, 10)),
        "p90_ratio": float(np.percentile(ratios, 90)),
        "mae_frames": float(mae),
        "zero_frame_ratio": float(zero_frames),
        "one_frame_ratio": float(one_frames),
    }


def main():
    # Load baseline
    ckpt = torch.load(str(BASELINE), map_location=device, weights_only=False)
    stats = ckpt["stats"]

    # Dataset
    dataset = TTSDataset(MANIFEST, FEAT_DIR, max_mel_len=2000)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # Compute mean log duration for reset
    all_dur = []
    for i in range(len(dataset)):
        all_dur.extend(dataset[i]["durations"].tolist())
    mean_dur = sum(all_dur) / len(all_dur)
    mean_log_dur = math.log(max(mean_dur, 1.0))
    print(f"Mean duration: {mean_dur:.1f} frames, log={mean_log_dur:.3f}")

    # ── D-warm ──
    print(f"\n{'='*60}")
    print(f"D-warm: keep step12000 duration predictor weights")
    print(f"{'='*60}")

    model_warm = FastSpeech2(vocab_size=268, dropout=0.0, mean_log_dur=mean_log_dur).to(device)
    model_warm.load_state_dict(ckpt["model"])
    # Don't reinit duration predictor — keep step12000 weights
    freeze_acoustic(model_warm)

    # Evaluate before training
    print(f"\nBefore training:")
    eval_warm_before = eval_duration(model_warm, dataset)
    print(f"  median ratio={eval_warm_before['median_ratio']:.1%} MAE={eval_warm_before['mae_frames']:.1f}")

    train_dur_only(model_warm, loader, steps=500, lr=5e-4, label="D-warm")

    print(f"\nAfter training:")
    eval_warm_after = eval_duration(model_warm, dataset)
    print(f"  median ratio={eval_warm_after['median_ratio']:.1%} mean={eval_warm_after['mean_ratio']:.1%}")
    print(f"  P10={eval_warm_after['p10_ratio']:.1%} P90={eval_warm_after['p90_ratio']:.1%}")
    print(f"  MAE={eval_warm_after['mae_frames']:.1f} frames")
    print(f"  0-frame: {eval_warm_after['zero_frame_ratio']:.1%} 1-frame: {eval_warm_after['one_frame_ratio']:.1%}")

    # Save
    torch.save({"model": model_warm.state_dict(), "stats": stats},
               str(OUT / "dur_dwarm.pt"))
    print(f"  Saved: dur_dwarm.pt")

    # ── D-reset ──
    print(f"\n{'='*60}")
    print(f"D-reset: reinitialize duration predictor")
    print(f"{'='*60}")

    model_reset = FastSpeech2(vocab_size=268, dropout=0.0, mean_log_dur=mean_log_dur).to(device)
    model_reset.load_state_dict(ckpt["model"])
    # Reinitialize duration predictor
    model_reset.duration_predictor.init_bias(mean_log_dur)
    freeze_acoustic(model_reset)

    print(f"\nBefore training:")
    eval_reset_before = eval_duration(model_reset, dataset)
    print(f"  median ratio={eval_reset_before['median_ratio']:.1%} MAE={eval_reset_before['mae_frames']:.1f}")

    train_dur_only(model_reset, loader, steps=500, lr=5e-4, label="D-reset")

    print(f"\nAfter training:")
    eval_reset_after = eval_duration(model_reset, dataset)
    print(f"  median ratio={eval_reset_after['median_ratio']:.1%} mean={eval_reset_after['mean_ratio']:.1%}")
    print(f"  P10={eval_reset_after['p10_ratio']:.1%} P90={eval_reset_after['p90_ratio']:.1%}")
    print(f"  MAE={eval_reset_after['mae_frames']:.1f} frames")
    print(f"  0-frame: {eval_reset_after['zero_frame_ratio']:.1%} 1-frame: {eval_reset_after['one_frame_ratio']:.1%}")

    torch.save({"model": model_reset.state_dict(), "stats": stats},
               str(OUT / "dur_dreset.pt"))
    print(f"  Saved: dur_dreset.pt")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"D-warm vs D-reset Summary (500 steps)")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'D-warm':>10} {'D-reset':>10}")
    print("-" * 40)
    for key in ["median_ratio", "mean_ratio", "p10_ratio", "p90_ratio", "mae_frames", "zero_frame_ratio", "one_frame_ratio"]:
        print(f"{key:<20} {eval_warm_after[key]:>10.1%} {eval_reset_after[key]:>10.1%}" if "ratio" in key else f"{key:<20} {eval_warm_after[key]:>10.1f} {eval_reset_after[key]:>10.1f}")


if __name__ == "__main__":
    main()
