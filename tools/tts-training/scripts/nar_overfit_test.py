"""Stage 3: Single-sample overfit test for FastSpeech 2.

Goal: Verify the model can memorize one training sample.
If loss drops to near-zero and the generated mel matches ground truth,
the data pipeline + model architecture are correct.

Usage:
    python nar_overfit_test.py [--poem_id poem_0000] [--steps 500] [--lr 1e-3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nar_fastspeech2 import FastSpeech2

MANIFEST = ROOT / "data" / "train_manifest.jsonl"


def load_sample(poem_id=None):
    """Load one sample from manifest."""
    records = []
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if poem_id:
        for r in records:
            if r["poem_id"] == poem_id:
                return r
        raise ValueError(f"poem_id {poem_id} not found")

    return records[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poem_id", default=None, help="Specific poem to overfit")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Steps: {args.steps}, LR: {args.lr}")

    # Pick a medium-length sample if not specified
    if args.poem_id is None:
        rec = load_sample()
        # Find a short-ish sample for faster overfit
        records = []
        with open(MANIFEST, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        records.sort(key=lambda r: r["mel_len"])
        rec = records[len(records) // 4]
    else:
        rec = load_sample(args.poem_id)

    print(f"\nSample: {rec['poem_id']}")
    print(f"  text: {rec['text'][:50]}")
    print(f"  n_phonemes: {rec['n_phonemes']}")
    print(f"  mel_len: {rec['mel_len']}")

    # Load features
    npz = np.load(str(ROOT / rec["mel_path"]))
    mel_gt = npz["mel"]
    f0_gt = npz["f0"]
    energy_gt = npz["energy"]
    durations_gt = np.array(rec["durations"], dtype=np.int32)

    print(f"  mel: {mel_gt.shape}, range=[{mel_gt.min():.2f}, {mel_gt.max():.2f}]")
    print(f"  f0: {f0_gt.shape}, voiced={np.sum(f0_gt > 0)}/{len(f0_gt)}")
    print(f"  energy: {energy_gt.shape}")
    print(f"  durations: {durations_gt.shape}, sum={durations_gt.sum()}")
    assert durations_gt.sum() == mel_gt.shape[0], "Duration invariant violated!"

    # Prepare tensors
    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long, device=args.device)
    durations = torch.tensor([rec["durations"]], dtype=torch.long, device=args.device)
    mel_target = torch.tensor(mel_gt, dtype=torch.float32, device=args.device).unsqueeze(0)

    # Normalize F0 (log domain, voiced only)
    voiced_mask = f0_gt > 0
    if voiced_mask.sum() > 0:
        f0_log = np.zeros_like(f0_gt)
        f0_log[voiced_mask] = np.log(f0_gt[voiced_mask])
        f0_mean = f0_log[voiced_mask].mean()
        f0_std = f0_log[voiced_mask].std() + 1e-8
        f0_norm = np.zeros_like(f0_gt)
        f0_norm[voiced_mask] = (f0_log[voiced_mask] - f0_mean) / f0_std
    else:
        f0_norm = np.zeros_like(f0_gt)
    pitches = torch.tensor(f0_norm, dtype=torch.float32, device=args.device).unsqueeze(0)

    # Normalize energy
    e_mean = energy_gt.mean()
    e_std = energy_gt.std() + 1e-8
    energy_norm = (energy_gt - e_mean) / e_std
    energies = torch.tensor(energy_norm, dtype=torch.float32, device=args.device).unsqueeze(0)

    # Create model
    model = FastSpeech2(
        vocab_size=268,
        d_model=256,
        nhead=2,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        n_mels=80,
        dropout=0.0,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {n_params:,} ({n_params / 1e6:.1f}M)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Overfit training")
    print(f"{'='*60}")

    best_loss = float("inf")
    best_mel_l1 = float("inf")

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()

        mel_out, log_dur_pred, pitch_pred, energy_pred = model(
            phone_ids,
            durations=durations,
            pitches=pitches,
            energies=energies,
        )

        T_pred = mel_out.shape[1]
        T_gt = mel_target.shape[1]
        T_min = min(T_pred, T_gt)

        # Mel loss (L1)
        mel_loss = F.l1_loss(mel_out[:, :T_min], mel_target[:, :T_min])

        # Duration loss (MSE on log duration)
        log_dur_gt = torch.log(durations.float().clamp(min=1))
        L = min(log_dur_pred.shape[1], log_dur_gt.shape[1])
        dur_loss = F.mse_loss(log_dur_pred[:, :L], log_dur_gt[:, :L])

        # Pitch loss (L1, only on expanded enc predictions)
        pitch_gt_expanded = pitches[:, :T_min]
        pitch_loss = F.l1_loss(pitch_pred[:, :T_min], pitch_gt_expanded[:, :T_min])

        # Energy loss
        energy_gt_expanded = energies[:, :T_min]
        energy_loss = F.l1_loss(energy_pred[:, :T_min], energy_gt_expanded[:, :T_min])

        total_loss = mel_loss + dur_loss + pitch_loss + energy_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == 1 or step == args.steps:
            print(
                f"  step {step:4d}: "
                f"total={total_loss.item():.4f} "
                f"mel={mel_loss.item():.4f} "
                f"dur={dur_loss.item():.4f} "
                f"pitch={pitch_loss.item():.4f} "
                f"energy={energy_loss.item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

        if mel_loss.item() < best_mel_l1:
            best_mel_l1 = mel_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\n  Best mel L1: {best_mel_l1:.4f}")

    # Inference test with predicted durations
    print(f"\n{'='*60}")
    print(f"Inference test (predicted duration)")
    print(f"{'='*60}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mel_pred, dur_pred, pitch_pred, energy_pred = model(phone_ids)

    print(f"  Predicted mel shape: {mel_pred.shape}")
    print(f"  Target mel shape: {mel_target.shape}")
    print(f"  Predicted duration sum: {dur_pred.exp().sum().item():.0f}")
    print(f"  Target duration sum: {durations.sum().item()}")

    # Mel L1 on inference
    T_pred = mel_pred.shape[1]
    T_gt = mel_target.shape[1]
    T_min = min(T_pred, T_gt)
    inf_l1 = F.l1_loss(mel_pred[:, :T_min], mel_target[:, :T_min]).item()
    print(f"  Inference mel L1: {inf_l1:.4f}")

    # Save predicted mel for vocoder test
    output_dir = ROOT / "output" / "overfit"
    output_dir.mkdir(parents=True, exist_ok=True)

    mel_save = mel_pred[0].cpu().numpy()
    np.save(str(output_dir / f"{rec['poem_id']}_mel_pred.npy"), mel_save)
    np.save(str(output_dir / f"{rec['poem_id']}_mel_gt.npy"), mel_gt)
    print(f"\n  Saved mel to {output_dir}")

    # Verdict
    print(f"\n{'='*60}")
    if best_mel_l1 < 0.3:
        print(f"RESULT: PASS — mel L1 {best_mel_l1:.4f} < 0.3 (model can overfit)")
    elif best_mel_l1 < 0.5:
        print(f"RESULT: MARGINAL — mel L1 {best_mel_l1:.4f} (needs more steps or higher LR)")
    else:
        print(f"RESULT: FAIL — mel L1 {best_mel_l1:.4f} > 0.5 (check data/model)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
