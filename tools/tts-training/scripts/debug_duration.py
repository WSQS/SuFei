"""Debug duration predictor behavior."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nar_fastspeech2 import FastSpeech2

MANIFEST = ROOT / "data" / "train_manifest.jsonl"

# Load sample
recs = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
rec = next(r for r in recs if r["poem_id"] == "poem_0285")

npz = np.load(str(ROOT / rec["mel_path"]))
mel_gt = npz["mel"]

device = "cuda"
phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long, device=device)
durations = torch.tensor([rec["durations"]], dtype=torch.long, device=device)
mel_target = torch.tensor(mel_gt, dtype=torch.float32, device=device).unsqueeze(0)

# Create fresh model
model = FastSpeech2(vocab_size=268, dropout=0.0).to(device)

# Quick 500-step training focusing on duration
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1, 501):
    model.train()
    optimizer.zero_grad()

    mel_out, log_dur_pred, _, _ = model(phone_ids, durations=durations)

    log_dur_gt = torch.log(durations.float().clamp(min=1))
    L = min(log_dur_pred.shape[1], log_dur_gt.shape[1])
    dur_loss = torch.nn.functional.mse_loss(log_dur_pred[:, :L], log_dur_gt[:, :L])

    # Also train mel
    T_pred = mel_out.shape[1]
    T_gt = mel_target.shape[1]
    T_min = min(T_pred, T_gt)
    mel_loss = torch.nn.functional.l1_loss(mel_out[:, :T_min], mel_target[:, :T_min])

    total = mel_loss + dur_loss
    total.backward()
    optimizer.step()

    if step % 100 == 0 or step == 1:
        # Check predicted durations
        pred_dur_exp = log_dur_pred.exp().detach()
        print(
            f"step {step}: "
            f"dur_loss={dur_loss.item():.4f} "
            f"mel_loss={mel_loss.item():.4f} "
            f"pred_dur_sum={pred_dur_exp.sum().item():.0f} "
            f"gt_dur_sum={durations.sum().item()} "
            f"pred_range=[{pred_dur_exp.min().item():.2f}, {pred_dur_exp.max().item():.2f}] "
            f"gt_range=[{durations.float().min().item():.0f}, {durations.float().max().item():.0f}]"
        )

# Check per-position durations
model.eval()
with torch.no_grad():
    _, log_dur_pred, _, _ = model(phone_ids)

pred_dur = log_dur_pred.exp()
gt_dur = durations.float()
print(f"\nPer-position durations (first 20):")
for i in range(min(20, len(rec["durations"]))):
    print(f"  pos {i}: gt={gt_dur[0,i].item():.0f} pred={pred_dur[0,i].item():.1f}")
