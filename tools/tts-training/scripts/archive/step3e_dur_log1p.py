"""Step 3e: Duration predictor with log1p/expm1 transform (no clamp).

Previous: log(clamp(dur, 1, 100)) → predictor can't exceed 100 frames
Fix: log1p(dur) → predictor can learn full range including punctuation
     Inference: expm1(pred) instead of exp(pred)

This is the correct fix for the punctuation pause problem.
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
    def __init__(self, d_model=256, bias=2.0):
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


def eval_duration(model, dur_model, dataset):
    """Evaluate using expm1 (correct inverse of log1p)."""
    model.eval()
    dur_model.eval()
    ratios = []
    for i in range(len(dataset)):
        item = dataset[i]
        ids = torch.tensor(np.array(item["phoneme_ids"]), device=device).unsqueeze(0)
        gt_dur = np.array(item["durations"])
        with torch.no_grad():
            x = model.embedding(ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x)
            log_dur = dur_model(x)
        # Use expm1 as inverse of log1p
        pred_dur = torch.expm1(log_dur[0]).cpu().numpy()
        pred_dur = np.maximum(pred_dur, 0)  # clamp negative to 0
        ratio = pred_dur.sum() / max(gt_dur.sum(), 1)
        ratios.append(ratio)
    return float(np.median(ratios)), float(np.mean(ratios))


def main():
    ckpt = torch.load(str(BASELINE), map_location=device, weights_only=False)
    stats = ckpt["stats"]

    dataset = TTSDataset(MANIFEST, FEAT_DIR, max_mel_len=2000)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # Compute bias using log1p (not log+clamp)
    all_dur = []
    for i in range(len(dataset)):
        all_dur.extend(dataset[i]["durations"].tolist())
    all_dur = np.array(all_dur, dtype=np.float64)
    log1p_durations = np.log1p(all_dur)
    correct_bias = float(log1p_durations.mean())
    correct_std = float(log1p_durations.std())

    print(f"log1p duration stats:")
    print(f"  mean(log1p(dur)) = {correct_bias:.3f}")
    print(f"  expm1({correct_bias:.3f}) = {math.expm1(correct_bias):.1f} frames")
    print(f"  std = {correct_std:.3f}")
    print(f"  Previous wrong: log(clamp(dur,1,100)) mean = 1.347")

    model = FastSpeech2(vocab_size=268, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dur_model = DurationModel(d_model=256, bias=correct_bias).to(device)
    dur_params = list(dur_model.parameters())
    print(f"\nDuration model params: {sum(p.numel() for p in dur_params):,}")
    optimizer = torch.optim.Adam(dur_params, lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    med, mean_r = eval_duration(model, dur_model, dataset)
    print(f"\nBefore: median ratio={med:.1%}, mean={mean_r:.1%}")

    step = 0
    t_start = time.time()
    max_steps = 5000

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

            # Use log1p transform (NO clamp)
            log_dur_gt = torch.log1p(durations_gt)
            L = min(log_dur_pred.size(1), log_dur_gt.size(1))
            dur_loss = masked_l1_loss(
                log_dur_pred[:, :L], log_dur_gt[:, :L], phone_mask[:, :L]
            )
            dur_loss.backward()
            torch.nn.utils.clip_grad_norm_(dur_model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % 500 == 0 or step == 1:
                med, mean_r = eval_duration(model, dur_model, dataset)
                elapsed = time.time() - t_start
                print(f"  step {step:4d}: dur_loss={dur_loss.item():.4f} median={med:.1%} mean={mean_r:.1%} ({elapsed:.0f}s)")

    print(f"\n{'='*60}")
    print(f"FINAL: median ratio={med:.1%}, mean ratio={mean_r:.1%}")
    print(f"{'='*60}")

    torch.save({"dur_model": dur_model.state_dict(), "stats": stats,
                "bias": correct_bias},
               str(OUT / "dur_log1p.pt"))
    print(f"Saved: dur_log1p.pt")


if __name__ == "__main__":
    main()
