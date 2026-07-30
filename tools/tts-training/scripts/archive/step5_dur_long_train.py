"""Step 5: Long duration training (15k steps) + per-phoneme boundary analysis.

Extends step3e with:
- 15k steps (was 5k)
- Cosine LR schedule
- Boundary error tracking every 1000 steps
- Final per-phoneme breakdown
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2, TransformerBlock
from nar_train import TTSDataset, collate_fn, masked_l1_loss

BASELINE = ROOT / "checkpoints" / "acoustic_baseline_step12000.pt"
MANIFEST = ROOT / "data" / "train_manifest.jsonl"
FEAT_DIR = ROOT / "data" / "nar_features"
OUT = ROOT / "checkpoints"

device = "cuda" if torch.cuda.is_available() else "cpu"


class DurationModel(torch.nn.Module):
    def __init__(self, d_model=256, bias=1.668):
        super().__init__()
        self.adapter = TransformerBlock(d_model, nhead=2, dim_feedforward=512, dropout=0.1)
        self.conv1 = torch.nn.Conv1d(d_model, 256, 3, padding=1)
        self.norm1 = torch.nn.LayerNorm(256)
        self.conv2 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self.norm2 = torch.nn.LayerNorm(256)
        self.linear = torch.nn.Linear(256, 1)
        self.dropout = torch.nn.Dropout(0.1)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=1e-4)
        torch.nn.init.constant_(self.linear.bias, bias)

    def forward(self, x, mask=None):
        x = self.adapter(x, mask=mask)
        x = x.transpose(1, 2)
        x = self.dropout(torch.relu(self.conv1(x)))
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.dropout(torch.relu(self.conv2(x)))
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.dropout(self.linear(x))
        return x.squeeze(-1)


def eval_duration(model, dur_model, dataset, n_samples=None):
    """Evaluate duration + boundary errors."""
    model.eval()
    dur_model.eval()
    ratios = []
    boundary_errors = []

    n = n_samples or len(dataset)
    for i in range(n):
        item = dataset[i]
        ids = torch.tensor(np.array(item["phoneme_ids"]), device=device).unsqueeze(0)
        gt_dur = np.array(item["durations"], dtype=np.float64)

        with torch.no_grad():
            x = model.embedding(ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x)
            log_dur = dur_model(x)

        pred_dur = torch.expm1(log_dur[0]).clamp(min=0).cpu().numpy()

        ratio = pred_dur.sum() / max(gt_dur.sum(), 1)
        ratios.append(ratio)

        # Boundary error: cumulative duration difference
        gt_cumsum = np.cumsum(gt_dur)
        pred_cumsum = np.cumsum(pred_dur)
        L = min(len(gt_cumsum), len(pred_cumsum))
        boundary_mae = np.abs(gt_cumsum[:L] - pred_cumsum[:L]).mean()
        boundary_errors.append(boundary_mae)

    ratios = np.array(ratios)
    boundary_errors = np.array(boundary_errors)
    return {
        "median_ratio": float(np.median(ratios)),
        "mean_ratio": float(ratios.mean()),
        "p10_ratio": float(np.percentile(ratios, 10)),
        "p90_ratio": float(np.percentile(ratios, 90)),
        "boundary_mae": float(np.mean(boundary_errors)),
        "boundary_p90": float(np.percentile(boundary_errors, 90)),
    }


def main():
    ckpt = torch.load(str(BASELINE), map_location=device, weights_only=False)
    stats = ckpt["stats"]

    dataset = TTSDataset(MANIFEST, FEAT_DIR, max_mel_len=2000)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)

    all_dur = np.array([d for i in range(len(dataset)) for d in dataset[i]["durations"].tolist()], dtype=np.float64)
    correct_bias = float(np.log1p(all_dur).mean())

    print(f"Bias: {correct_bias:.3f} (expm1={math.expm1(correct_bias):.1f})")

    model = FastSpeech2(vocab_size=268, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dur_model = DurationModel(d_model=256, bias=correct_bias).to(device)
    optimizer = torch.optim.Adam(dur_model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    max_steps = 15000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)

    step = 0
    t_start = time.time()

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            dur_model.train()
            optimizer.zero_grad()

            phoneme_ids = batch["phoneme_ids"].to(device)
            durations_gt = batch["durations"].to(device).float()
            phone_mask = batch["phone_mask"].to(device)

            with torch.no_grad():
                x = model.embedding(phoneme_ids) * math.sqrt(model.d_model)
                x = model.pos_enc(x)
                for layer in model.encoder_layers:
                    x = layer(x, mask=phone_mask)

            log_dur_pred = dur_model(x, mask=phone_mask)
            log_dur_gt = torch.log1p(durations_gt)
            L = min(log_dur_pred.size(1), log_dur_gt.size(1))
            dur_loss = masked_l1_loss(log_dur_pred[:, :L], log_dur_gt[:, :L], phone_mask[:, :L])
            dur_loss.backward()
            torch.nn.utils.clip_grad_norm_(dur_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 1000 == 0 or step == 1:
                metrics = eval_duration(model, dur_model, dataset, n_samples=30)
                elapsed = time.time() - t_start
                print(f"step {step:5d}: loss={dur_loss.item():.4f} "
                      f"ratio={metrics['median_ratio']:.0%}/{metrics['mean_ratio']:.0%} "
                      f"boundary_mae={metrics['boundary_mae']:.1f}f "
                      f"lr={scheduler.get_last_lr()[0]:.6f} "
                      f"({elapsed:.0f}s)")

                # Save checkpoint every 5000 steps
                if step % 5000 == 0:
                    torch.save({"dur_model": dur_model.state_dict(), "stats": stats,
                                "bias": correct_bias, "step": step},
                               str(OUT / f"dur_log1p_step{step}.pt"))
                    print(f"  Saved: dur_log1p_step{step}.pt")

    # Final eval (all 204 samples)
    print(f"\n{'='*60}")
    metrics = eval_duration(model, dur_model, dataset)
    print(f"Final (all {len(dataset)} samples):")
    print(f"  median ratio: {metrics['median_ratio']:.1%}")
    print(f"  mean ratio:   {metrics['mean_ratio']:.1%}")
    print(f"  P10 ratio:    {metrics['p10_ratio']:.1%}")
    print(f"  P90 ratio:    {metrics['p90_ratio']:.1%}")
    print(f"  boundary MAE: {metrics['boundary_mae']:.1f} frames")
    print(f"  boundary P90: {metrics['boundary_p90']:.1f} frames")

    torch.save({"dur_model": dur_model.state_dict(), "stats": stats,
                "bias": correct_bias, "step": step},
               str(OUT / "dur_log1p_final.pt"))
    print(f"Saved: dur_log1p_final.pt")


if __name__ == "__main__":
    main()
