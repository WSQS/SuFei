"""Step 3c: Duration predictor with independent adapter.

Adds a private 1-layer Transformer block between encoder and duration predictor.
This gives the duration predictor its own feature transformation path,
independent of the mel-optimized encoder features.

Architecture change:
  encoder → shared features
    ├── decoder (mel)
    └── duration_adapter → duration_predictor  ← NEW

The duration_adapter is a TransformerBlock that transforms encoder features
into a duration-friendly representation before feeding to the VariancePredictor.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


class DurationModel(nn.Module):
    """Duration predictor with private adapter + variance predictor."""

    def __init__(self, d_model=256, mean_log_dur=2.7):
        super().__init__()
        self.adapter = TransformerBlock(d_model, nhead=2, dim_feedforward=512, dropout=0.1)
        self.predictor_conv1 = nn.Conv1d(d_model, 256, 3, padding=1)
        self.norm1 = nn.LayerNorm(256)
        self.predictor_conv2 = nn.Conv1d(256, 256, 3, padding=1)
        self.norm2 = nn.LayerNorm(256)
        self.linear = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.1)

        # Init: tiny weight + correct bias
        nn.init.normal_(self.linear.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.linear.bias, mean_log_dur)

    def forward(self, x, mask=None):
        # Private adapter
        x = self.adapter(x, mask=mask)
        # Conv predictor
        x = x.transpose(1, 2)
        x = self.dropout(torch.relu(self.predictor_conv1(x)))
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.dropout(torch.relu(self.predictor_conv2(x)))
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.dropout(self.linear(x))
        return x.squeeze(-1)


def eval_duration(model, dur_model, dataset):
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
        ratio = log_dur[0].exp().sum().item() / max(gt_dur.sum(), 1)
        ratios.append(ratio)
    return float(np.median(ratios)), float(np.mean(ratios))


def main():
    ckpt = torch.load(str(BASELINE), map_location=device, weights_only=False)
    stats = ckpt["stats"]

    dataset = TTSDataset(MANIFEST, FEAT_DIR, max_mel_len=2000)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)

    all_dur = []
    for i in range(len(dataset)):
        all_dur.extend(dataset[i]["durations"].tolist())
    mean_dur = sum(all_dur) / len(all_dur)
    mean_log_dur = math.log(max(mean_dur, 1.0))

    # Load frozen acoustic model
    model = FastSpeech2(vocab_size=268, dropout=0.0, mean_log_dur=mean_log_dur).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create independent duration model
    dur_model = DurationModel(d_model=256, mean_log_dur=mean_log_dur).to(device)
    dur_params = list(dur_model.parameters())
    print(f"Duration model params: {sum(p.numel() for p in dur_params):,}")
    optimizer = torch.optim.Adam(dur_params, lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    # Eval before
    med, mean_r = eval_duration(model, dur_model, dataset)
    print(f"Before: median ratio={med:.1%}, mean={mean_r:.1%}")

    # Train
    step = 0
    t_start = time.time()
    max_steps = 3000

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            dur_model.train()
            optimizer.zero_grad()

            phoneme_ids = batch["phoneme_ids"].to(device)
            durations_gt = batch["durations"].to(device)
            phone_mask = batch["phone_mask"].to(device)

            with torch.no_grad():
                x = model.embedding(phoneme_ids) * math.sqrt(model.d_model)
                x = model.pos_enc(x)
                for layer in model.encoder_layers:
                    x = layer(x, mask=phone_mask)

            log_dur_pred = dur_model(x, mask=phone_mask)

            dur_clipped = durations_gt.float().clamp(min=1, max=100)
            log_dur_gt = torch.log(dur_clipped)
            L = min(log_dur_pred.size(1), log_dur_gt.size(1))
            dur_loss = masked_l1_loss(
                log_dur_pred[:, :L], log_dur_gt[:, :L], phone_mask[:, :L]
            )
            dur_loss.backward()
            torch.nn.utils.clip_grad_norm_(dur_model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % 300 == 0 or step == 1:
                med, mean_r = eval_duration(model, dur_model, dataset)
                elapsed = time.time() - t_start
                print(f"  step {step:4d}: dur_loss={dur_loss.item():.4f} median={med:.1%} mean={mean_r:.1%} ({elapsed:.0f}s)")

    # Final
    print(f"\n{'='*60}")
    print(f"Final: median ratio={med:.1%}, mean ratio={mean_r:.1%}")

    # Save
    torch.save({"dur_model": dur_model.state_dict(), "stats": stats,
                "mean_log_dur": mean_log_dur},
               str(OUT / "dur_adapter.pt"))
    print(f"Saved: dur_adapter.pt")


if __name__ == "__main__":
    main()
