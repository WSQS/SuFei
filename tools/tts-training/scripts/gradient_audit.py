"""Measure gradient norms and cosine for mel vs duration on shared encoder."""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2
from nar_train import TTSDataset, collate_fn, length_regulate_batch, masked_l1_loss

MANIFEST = ROOT / "data" / "train_manifest.jsonl"
FEAT_DIR = ROOT / "data" / "nar_features"

device = "cuda"

# Load norm stats
import json
with open(ROOT / "data" / "norm_stats.json") as f:
    stats = json.load(f)
mel_mean = torch.tensor(stats["mel_mean"], device=device)
mel_std = torch.tensor(stats["mel_std"], device=device)
f0_mean = stats["f0_mean"]
f0_std = stats["f0_std"]
energy_mean = stats["energy_mean"]
energy_std = stats["energy_std"]

# Dataset + batch
dataset = TTSDataset(MANIFEST, FEAT_DIR, max_mel_len=2000)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
batch = next(iter(loader))

phoneme_ids = batch["phoneme_ids"].to(device)
durations_gt = batch["durations"].to(device)
mel_gt = batch["mel"].to(device)
f0_gt = batch["f0"].to(device)
energy_gt = batch["energy"].to(device)
phone_mask = batch["phone_mask"].to(device)
mel_mask = batch["mel_mask"].to(device)

# Normalize
mel_norm = (mel_gt - mel_mean) / mel_std
f0_norm = torch.where(f0_gt > 0, (torch.log(f0_gt.clamp(min=1)) - f0_mean) / f0_std, torch.zeros_like(f0_gt))
energy_norm = (energy_gt - energy_mean) / energy_std


def flat_grads(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    parts = [g.detach().flatten() for g in grads if g is not None]
    return torch.cat(parts) if parts else torch.zeros(0)


# Fresh model with zero-weight init
model = FastSpeech2(vocab_size=268, dropout=0.0, mean_log_dur=2.7).to(device)

# Forward pass
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

pitch_expanded = length_regulate_batch(pitch_pred_enc.unsqueeze(-1), durations_gt).squeeze(-1)
energy_expanded = length_regulate_batch(energy_pred_enc.unsqueeze(-1), durations_gt).squeeze(-1)

# Use predicted pitch/energy for decoder (round 6 approach)
pitch_embed = model.pitch_embed(pitch_expanded[:, :T_pred].unsqueeze(-1))
energy_embed = model.energy_embed(energy_expanded[:, :T_pred].unsqueeze(-1))
mel_input = mel_input + pitch_embed + energy_embed
mel_input = model.pos_enc(mel_input)
dec = mel_input
for layer in model.decoder_layers:
    dec = layer(dec)
mel_pred = model.mel_linear(dec)

# Compute losses
mel_loss = masked_l1_loss(mel_pred[:, :T_min], mel_norm[:, :T_min], mel_mask[:, :T_min])
dur_clipped = durations_gt.float().clamp(min=1, max=100)
log_dur_gt = torch.log(dur_clipped)
L_phone = min(log_dur_pred.size(1), log_dur_gt.size(1))
dur_loss = masked_l1_loss(log_dur_pred[:, :L_phone], log_dur_gt[:, :L_phone], phone_mask[:, :L_phone])

print(f"mel_loss: {mel_loss.item():.4f} (shape: {mel_pred[:,:T_min].shape})")
print(f"dur_loss: {dur_loss.item():.4f} (shape: {log_dur_pred[:,:L_phone].shape})")

# Check loss gradients
encoder_params = [p for p in model.encoder_layers.parameters() if p.requires_grad]
dur_head_params = [p for p in model.duration_predictor.parameters() if p.requires_grad]

# Test with different w_dur values
for w_dur in [1.0, 5.0, 10.0, 50.0]:
    # Need fresh grads
    model.zero_grad()
    
    # Recompute (without mel loss's backward interference)
    # Actually, let me compute grads separately for each loss
    
    # Mel grads
    model.zero_grad()
    mel_loss = masked_l1_loss(mel_pred[:, :T_min], mel_norm[:, :T_min], mel_mask[:, :T_min])
    g_mel_enc = flat_grads(mel_loss, encoder_params)
    
    # Dur grads
    model.zero_grad()
    dur_loss = masked_l1_loss(log_dur_pred[:, :L_phone], log_dur_gt[:, :L_phone], phone_mask[:, :L_phone])
    g_dur_enc = flat_grads(w_dur * dur_loss, encoder_params)
    g_dur_head = flat_grads(w_dur * dur_loss, dur_head_params)
    
    mel_norm_val = g_mel_enc.norm().item()
    dur_norm_val = g_dur_enc.norm().item()
    dur_head_norm_val = g_dur_head.norm().item()
    ratio = mel_norm_val / max(dur_norm_val, 1e-12)
    
    if g_mel_enc.numel() > 0 and g_dur_enc.numel() > 0:
        cosine = torch.dot(g_mel_enc, g_dur_enc) / (mel_norm_val * max(dur_norm_val, 1e-12) + 1e-12)
    else:
        cosine = 0.0
    
    print(f"\nw_dur={w_dur}:")
    print(f"  encoder mel grad norm: {mel_norm_val:.2f}")
    print(f"  encoder dur grad norm: {dur_norm_val:.4f}")
    print(f"  mel/dur ratio: {ratio:.1f}")
    print(f"  gradient cosine: {cosine:.4f}")
    print(f"  duration head grad norm: {dur_head_norm_val:.4f}")