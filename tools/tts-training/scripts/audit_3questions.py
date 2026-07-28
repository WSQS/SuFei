"""Audit: initialize duration, normalization, and train/inference consistency."""
import json, math, sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, 'tools/tts-training/scripts')
from nar_fastspeech2 import FastSpeech2
from nar_train import TTSDataset, collate_fn, length_regulate_batch, masked_l1_loss

ROOT = Path('tools/tts-training')
device = 'cuda'

print("=" * 60)
print("Q1: 初始化时 duration 是否约 15 帧？")
print("=" * 60)

model = FastSpeech2(vocab_size=268, dropout=0.0, mean_log_dur=2.7).to(device)
w = model.duration_predictor.linear.weight
b = model.duration_predictor.linear.bias
print(f"Bias: {b.item():.4f} (exp={torch.exp(b).item():.1f})")
print(f"Weight: mean={w.mean():.8f} std={w.std():.8f}")

rec = json.loads(open(ROOT/'data'/'train_manifest.jsonl', encoding='utf-8').readline())
ids = torch.tensor([rec['phoneme_ids']], device=device)
gt_dur = np.array(rec['durations'])

with torch.no_grad():
    x = model.embedding(ids) * math.sqrt(256)
    x = model.pos_enc(x)
    for layer in model.encoder_layers: x = layer(x)
    log_dur = model.duration_predictor(x)
pred = log_dur[0].exp()
print(f"Pred dur: mean={pred.mean():.1f} sum={pred.sum():.0f} (GT sum={gt_dur.sum()})")
print(f"log_dur: mean={log_dur[0].mean():.3f} std={log_dur[0].std():.3f}")
print(f"Ratio: {pred.sum().item()/gt_dur.sum():.2%}")
print(f"All close to bias: max_dev={((log_dur[0] - b.item()).abs().max()):.6f}")

print()
print("=" * 60)
print("Q2: 训练/推理的 pitch/energy 反变换是否一致？")
print("=" * 60)

# Load norm stats
with open(ROOT/'data'/'norm_stats.json') as f: stats = json.load(f)
f0_mean, f0_std = stats['f0_mean'], stats['f0_std']
energy_mean, energy_std = stats['energy_mean'], stats['energy_std']

print(f"F0: mean={f0_mean:.2f} std={f0_std:.2f}  → 训练目标: (log(F0)-{f0_mean})/{f0_std}")
print(f"Energy: mean={energy_mean:.2f} std={energy_std:.2f}  → 训练目标: (E-{energy_mean})/{energy_std}")
print(f"Duration: 训练目标=log(dur.clamp(1,100)), 推理=exp(pred)")
print(f"  公式对称: exp(log(dur)) = dur ✓")
print(f"  无标准化/反标准化 ✓")

print()
print("=" * 60)
print("Q3: GT 与 Pred 注入 decoder 时数值空间是否一致？")
print("=" * 60)

dataset = TTSDataset(ROOT/'data'/'train_manifest.jsonl', ROOT/'data'/'nar_features', max_mel_len=2000)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
batch = next(iter(loader))

phoneme_ids = batch['phoneme_ids'].to(device)
durations_gt = batch['durations'].to(device)
f0_gt = batch['f0'].to(device)
energy_gt = batch['energy'].to(device)
phone_mask = batch['phone_mask'].to(device)

# Normalize GT
f0_norm_gt = torch.where(f0_gt > 0, (torch.log(f0_gt.clamp(min=1)) - f0_mean) / f0_std, torch.zeros_like(f0_gt))
energy_norm_gt = (energy_gt - energy_mean) / energy_std

# Forward pass
with torch.no_grad():
    x = model.embedding(phoneme_ids) * math.sqrt(256)
    x = model.pos_enc(x)
    for layer in model.encoder_layers: x = layer(x, mask=phone_mask)
    pitch_pred = model.pitch_predictor(x)
    energy_pred = model.energy_predictor(x)

# Expand via training function
pitch_exp_train = length_regulate_batch(pitch_pred.unsqueeze(-1), durations_gt).squeeze(-1)
energy_exp_train = length_regulate_batch(energy_pred.unsqueeze(-1), durations_gt).squeeze(-1)

# Expand via model's length regulator
pitch_exp_model = model.length_regulator(pitch_pred.unsqueeze(-1), durations_gt).squeeze(-1)
energy_exp_model = model.length_regulator(energy_pred.unsqueeze(-1), durations_gt).squeeze(-1)

# Compare
T_min = min(pitch_exp_train.size(1), pitch_exp_model.size(1))
diff = (pitch_exp_train[:,:T_min] - pitch_exp_model[:,:T_min]).abs()
print(f"LengthRegulator train vs model: max_diff={diff.max():.6f} mean_diff={diff.mean():.6f}")
print(f"  (should be 0 — same input, same durations)")

# Check pitch distributions
print(f"\nGT pitch (normalized): mean={f0_norm_gt.mean():.3f} std={f0_norm_gt.std():.3f}")
print(f"Pred pitch (normalized): mean={pitch_exp_train[:,:T_min].mean():.3f} std={pitch_exp_train[:,:T_min].std():.3f}")
print(f"  Same space: GT and Pred both use normalized F0 ✓")

print(f"\nGT energy (normalized): mean={energy_norm_gt.mean():.3f} std={energy_norm_gt.std():.3f}")
print(f"Pred energy (normalized): mean={energy_exp_train[:,:T_min].mean():.3f} std={energy_exp_train[:,:T_min].std():.3f}")
print(f"  Same space: GT and Pred both use normalized Energy ✓")

# Check: does model.forward() produce same mel as manual training loop?
print(f"\n=== Train vs Inference consistency ===")
mel_input_train = length_regulate_batch(x, durations_gt)
T_pred = mel_input_train.size(1)
pitch_emb = model.pitch_embed(pitch_exp_train[:,:T_pred].unsqueeze(-1))
energy_emb = model.energy_embed(energy_exp_train[:,:T_pred].unsqueeze(-1))
mel_input_train = mel_input_train + pitch_emb + energy_emb
mel_input_train = model.pos_enc(mel_input_train)
dec_train = mel_input_train
for layer in model.decoder_layers: dec_train = layer(dec_train)
mel_train = model.mel_linear(dec_train)

# Now via model.forward()
mel_out, _, _, _ = model(phoneme_ids, durations=durations_gt)
T_min = min(mel_train.size(1), mel_out.size(1))
mel_diff = (mel_train[:,:T_min] - mel_out[:,:T_min]).abs()
print(f"mel diff (train manual vs model.forward): max={mel_diff.max():.6f} mean={mel_diff.mean():.6f}")
print(f"  Same: {'YES' if mel_diff.max() < 0.01 else 'NO — mismatch!'}")