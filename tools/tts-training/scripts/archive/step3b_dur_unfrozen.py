"""Step 3b: Duration predictor with unfrozen encoder.

Unfreezes encoder + duration predictor, keeps decoder frozen.
Uses separate optimizer groups: encoder (low LR) + duration (normal LR).
Tests on 16 samples first, then full dataset.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2
from nar_train import TTSDataset, collate_fn, masked_l1_loss

BASELINE = ROOT / "checkpoints" / "acoustic_baseline_step12000.pt"
MANIFEST = ROOT / "data" / "train_manifest.jsonl"
FEAT_DIR = ROOT / "data" / "nar_features"
OUT = ROOT / "checkpoints"

device = "cuda" if torch.cuda.is_available() else "cpu"


def eval_duration(model, dataset):
    model.eval()
    ratios = []
    for i in range(len(dataset)):
        item = dataset[i]
        ids = torch.tensor([np.array(item["phoneme_ids"])], device=device)
        gt_dur = np.array(item["durations"])
        with torch.no_grad():
            x = model.embedding(ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x)
            log_dur = model.duration_predictor(x)
        pred_sum = log_dur[0].exp().sum().item()
        ratio = pred_sum / max(gt_dur.sum(), 1)
        ratios.append(ratio)
    ratios = np.array(ratios)
    return float(np.median(ratios)), float(np.mean(ratios))


def main():
    ckpt = torch.load(str(BASELINE), map_location=device, weights_only=False)
    stats = ckpt["stats"]

    dataset = TTSDataset(MANIFEST, FEAT_DIR, max_mel_len=2000)

    # Compute mean log duration
    all_dur = []
    for i in range(len(dataset)):
        all_dur.extend(dataset[i]["durations"].tolist())
    mean_dur = sum(all_dur) / len(all_dur)
    mean_log_dur = math.log(max(mean_dur, 1.0))

    # ── 16-sample overfit ──
    print("=" * 60)
    print("Phase 1: 16-sample overfit (1000 steps)")
    print("=" * 60)

    subset_idx = list(range(0, 16))
    subset = Subset(dataset, subset_idx)
    loader = DataLoader(subset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model = FastSpeech2(vocab_size=268, dropout=0.0, mean_log_dur=mean_log_dur).to(device)
    model.load_state_dict(ckpt["model"])
    model.duration_predictor.init_bias(mean_log_dur)

    # Freeze decoder only, keep encoder + dur trainable
    for name, param in model.named_parameters():
        if any(k in name for k in ["decoder_layers", "mel_linear", "pitch_embed", "energy_embed"]):
            param.requires_grad = False
        elif "pitch_predictor" in name or "energy_predictor" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    enc_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and "encoder" in n]
    dur_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and "duration_predictor" in n]
    emb_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and "embedding" in n]

    print(f"Encoder params: {sum(p.numel() for p in enc_params):,}")
    print(f"Duration params: {sum(p.numel() for p in dur_params):,}")
    print(f"Embedding params: {sum(p.numel() for p in emb_params):,}")

    optimizer = torch.optim.Adam([
        {"params": enc_params, "lr": 1e-5},
        {"params": emb_params, "lr": 1e-5},
        {"params": dur_params, "lr": 5e-4},
    ])

    # Eval before
    med, mean = eval_duration(model, subset)
    print(f"\nBefore: median ratio={med:.1%}, mean={mean:.1%}")

    for step in range(1, 1001):
        for batch in loader:
            model.train()
            optimizer.zero_grad()

            phoneme_ids = batch["phoneme_ids"].to(device)
            durations_gt = batch["durations"].to(device)
            phone_mask = batch["phone_mask"].to(device)

            x = model.embedding(phoneme_ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x, mask=phone_mask)

            log_dur_pred = model.duration_predictor(x)

            dur_clipped = durations_gt.float().clamp(min=1, max=100)
            log_dur_gt = torch.log(dur_clipped)
            L = min(log_dur_pred.size(1), log_dur_gt.size(1))
            dur_loss = masked_l1_loss(
                log_dur_pred[:, :L], log_dur_gt[:, :L], phone_mask[:, :L]
            )
            dur_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 200 == 0 or step == 1:
                med, mean_r = eval_duration(model, subset)
                print(f"  step {step:4d}: dur_loss={dur_loss.item():.4f} median_ratio={med:.1%} mean_ratio={mean_r:.1%}")

    # ─�� Full dataset training ──
    print(f"\n{'='*60}")
    print("Phase 2: Full 204-sample training (3000 steps)")
    print("=" * 60)

    full_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # Reset optimizer for full training
    optimizer = torch.optim.Adam([
        {"params": enc_params, "lr": 1e-5},
        {"params": emb_params, "lr": 1e-5},
        {"params": dur_params, "lr": 5e-4},
    ])

    step = 0
    t_start = time.time()
    while step < 3000:
        for batch in full_loader:
            if step >= 3000:
                break
            model.train()
            optimizer.zero_grad()

            phoneme_ids = batch["phoneme_ids"].to(device)
            durations_gt = batch["durations"].to(device)
            phone_mask = batch["phone_mask"].to(device)

            x = model.embedding(phoneme_ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x, mask=phone_mask)

            log_dur_pred = model.duration_predictor(x)

            dur_clipped = durations_gt.float().clamp(min=1, max=100)
            log_dur_gt = torch.log(dur_clipped)
            L = min(log_dur_pred.size(1), log_dur_gt.size(1))
            dur_loss = masked_l1_loss(
                log_dur_pred[:, :L], log_dur_gt[:, :L], phone_mask[:, :L]
            )
            dur_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % 500 == 0 or step == 1:
                med, mean_r = eval_duration(model, dataset)
                elapsed = time.time() - t_start
                print(f"  step {step:4d}: dur_loss={dur_loss.item():.4f} median_ratio={med:.1%} mean_ratio={mean_r:.1%} ({elapsed:.0f}s)")

    # Final eval
    print(f"\n{'='*60}")
    print("Final Duration Evaluation (all 204 samples)")
    print(f"{'='*60}")
    med, mean_r = eval_duration(model, dataset)
    print(f"  median ratio: {med:.1%}")
    print(f"  mean ratio: {mean_r:.1%}")

    # Save
    torch.save({"model": model.state_dict(), "stats": stats, "step": step},
               str(OUT / "dur_unfrozen_encoder.pt"))
    print(f"  Saved: dur_unfrozen_encoder.pt")


if __name__ == "__main__":
    main()
