"""Deep analysis: why does pred-var fail on training data?

The model (D300fixE2E_step24000) was trained with GT durations for
length regulator, but pred-var inference uses predicted durations.
This creates a train/inference gap.

This script traces exactly where the mel degradation comes from:
1. Duration misalignment: how much do phoneme boundaries shift?
2. Pitch/energy mismatch: predicted vs GT at each frame
3. Mel error distribution: where are the worst frames?
"""
import json, sys, math, numpy as np, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nar_fastspeech2 import FastSpeech2

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "paddle_distill_features"
STATS = json.load(open(ROOT / "data" / "train_300_norm_stats.json"))
mel_mean = np.array(STATS["mel_mean"], dtype=np.float32)
mel_std = np.array(STATS["mel_std"], dtype=np.float32)

ckpt = torch.load(str(ROOT / "checkpoints" / "D300fixE2E_step24000_slim.pt"),
                   map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()

# Load phone map
phone_map = {}
with open(ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0" / "phone_id_map.txt",
          encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            phone_map[parts[0]] = int(parts[1])
id2phone = {v: k for k, v in phone_map.items()}

rec = [json.loads(l) for l in open(ROOT / "data" / "train_300_manifest.jsonl", encoding="utf-8")][0]
pid = rec["poem_id"]
npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
mel_gt = torch.tensor(npz["mel"].astype(np.float32)).unsqueeze(0)
f0_gt = torch.tensor(npz["f0"].astype(np.float32)).unsqueeze(0)
e_gt = torch.tensor(npz["energy"].astype(np.float32)).unsqueeze(0)
dur_gt = torch.tensor([rec["durations"]], dtype=torch.long)
phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)

mel_norm_gt = (mel_gt - torch.tensor(STATS["mel_mean"])) / torch.tensor(STATS["mel_std"])
f0_norm = torch.where(f0_gt > 0,
    (torch.log(f0_gt.clamp(min=1)) - STATS["f0_mean"]) / STATS["f0_std"],
    torch.zeros_like(f0_gt))
e_norm = (e_gt - STATS["energy_mean"]) / STATS["energy_std"]

phonemes = rec["phoneme_ids"]
gt_durs = rec["durations"]

print(f"Poem: {pid} — {rec['text'][:40]}")
print(f"Phonemes: {len(phonemes)}, GT dur total: {sum(gt_durs)}, mel frames: {mel_gt.shape[1]}")

# ── Step 1: Run encoder once ──
with torch.no_grad():
    x = model.embedding(phone_ids) * math.sqrt(model.d_model)
    x = model.pos_enc(x)
    for layer in model.encoder_layers:
        x = layer(x)
    enc_out = x

    log_dur_pred = model.duration_predictor(enc_out)
    pitch_pred = model.pitch_predictor(enc_out)
    energy_pred = model.energy_predictor(enc_out)

pred_durs = np.maximum(np.round(np.exp(log_dur_pred[0].numpy())), 0).astype(int)
pred_dur_t = torch.tensor([pred_durs], dtype=torch.long)

# ── Step 2: Analyze phoneme boundary alignment ──
print(f"\n{'='*70}")
print("1. PHONEME BOUNDARY ALIGNMENT")
print(f"{'='*70}")

gt_boundaries = np.cumsum(gt_durs)
pred_boundaries = np.cumsum(pred_durs)

print(f"\n{'Phone':>6} {'GT_dur':>6} {'Pred_dur':>8} {'GT_end':>6} {'Pred_end':>8} {'Shift':>6}")
print("-" * 42)
for i in range(min(20, len(phonemes))):
    pid_val = phonemes[i]
    name = id2phone.get(pid_val, f"?{pid_val}")
    shift = pred_boundaries[i] - gt_boundaries[i] if i < len(pred_durs) else 0
    print(f"{name:>6} {gt_durs[i]:>6} {pred_durs[i]:>8} "
          f"{gt_boundaries[i]:>6} {pred_boundaries[i] if i < len(pred_durs) else 0:>8} {shift:>+6}")

# ── Step 3: Run 3 decoder configs and compare ──
print(f"\n{'='*70}")
print("2. DECODER OUTPUT COMPARISON (3 configs)")
print(f"{'='*70}")

with torch.no_grad():
    # Config A: GT dur + GT pitch/energy (teacher forcing)
    mi_a = model.length_regulator(enc_out, dur_gt)
    T_a = mi_a.size(1)
    mi_a = mi_a + model.pitch_embed(f0_norm[:, :T_a].unsqueeze(-1)) + \
                  model.energy_embed(e_norm[:, :T_a].unsqueeze(-1))
    mi_a = model.pos_enc(mi_a)
    dec_a = mi_a
    for layer in model.decoder_layers:
        dec_a = layer(dec_a)
    mel_a = model.mel_linear(dec_a)  # [1, T_gt, 80]

    # Config B: GT dur + PREDICTED pitch/energy (training E2E mode)
    pitch_b = model.length_regulator(pitch_pred.unsqueeze(-1), dur_gt).squeeze(-1)
    energy_b = model.length_regulator(energy_pred.unsqueeze(-1), dur_gt).squeeze(-1)
    mi_b = model.length_regulator(enc_out, dur_gt)
    T_b = mi_b.size(1)
    mi_b = mi_b + model.pitch_embed(pitch_b[:, :T_b].unsqueeze(-1)) + \
                  model.energy_embed(energy_b[:, :T_b].unsqueeze(-1))
    mi_b = model.pos_enc(mi_b)
    dec_b = mi_b
    for layer in model.decoder_layers:
        dec_b = layer(dec_b)
    mel_b = model.mel_linear(dec_b)

    # Config C: PRED dur + PREDICTED pitch/energy (full pred-var = inference)
    pitch_c = model.length_regulator(pitch_pred.unsqueeze(-1), pred_dur_t).squeeze(-1)
    energy_c = model.length_regulator(energy_pred.unsqueeze(-1), pred_dur_t).squeeze(-1)
    mi_c = model.length_regulator(enc_out, pred_dur_t)
    T_c = mi_c.size(1)
    mi_c = mi_c + model.pitch_embed(pitch_c[:, :T_c].unsqueeze(-1)) + \
                  model.energy_embed(energy_c[:, :T_c].unsqueeze(-1))
    mi_c = model.pos_enc(mi_c)
    dec_c = mi_c
    for layer in model.decoder_layers:
        dec_c = layer(dec_c)
    mel_c = model.mel_linear(dec_c)

# Compare mel L1 for each config
T_cmp = min(mel_a.size(1), mel_b.size(1), mel_norm_gt.size(1))
l1_a = (mel_a[:, :T_cmp] - mel_norm_gt[:, :T_cmp]).abs().mean().item()
l1_b = (mel_b[:, :T_cmp] - mel_norm_gt[:, :T_cmp]).abs().mean().item()
T_c_cmp = min(mel_c.size(1), mel_norm_gt.size(1))
l1_c = (mel_c[:, :T_c_cmp] - mel_norm_gt[:, :T_c_cmp]).abs().mean().item()

print(f"\nConfig A (GT dur + GT pitch/energy):  mel L1 = {l1_a:.4f}")
print(f"Config B (GT dur + Pred pitch/energy): mel L1 = {l1_b:.4f}")
print(f"Config C (Pred dur + Pred pitch/energy): mel L1 = {l1_c:.4f}")
print(f"\nIsolate effects:")
print(f"  Pitch/energy prediction error (B-A): {l1_b - l1_a:+.4f}")
print(f"  Duration prediction error (C-B):     {l1_c - l1_b:+.4f}")

# ── Step 4: Pitch analysis per frame ──
print(f"\n{'='*70}")
print("3. PITCH/ENERGY: GT per-frame vs Predicted per-phone-expanded")
print(f"{'='*70}")

# GT pitch is per-frame. Predicted pitch is per-phone, expanded by dur.
# How well do they match?
f0_norm_np = f0_norm[0].numpy()
pitch_pred_expanded_gt = model.length_regulator(pitch_pred.unsqueeze(-1), dur_gt).squeeze(-1)[0].numpy()
pitch_pred_expanded_pred = model.length_regulator(pitch_pred.unsqueeze(-1), pred_dur_t).squeeze(-1)[0].numpy()

T_p = min(len(f0_norm_np), len(pitch_pred_expanded_gt))
pitch_l1_gt_expand = np.abs(pitch_pred_expanded_gt[:T_p] - f0_norm_np[:T_p]).mean()
T_pp = min(len(f0_norm_np), len(pitch_pred_expanded_pred))
pitch_l1_pred_expand = np.abs(pitch_pred_expanded_pred[:T_pp] - f0_norm_np[:T_pp]).mean()

print(f"\nPitch L1 (pred expanded by GT dur vs GT per-frame):   {pitch_l1_gt_expand:.4f}")
print(f"Pitch L1 (pred expanded by Pred dur vs GT per-frame):  {pitch_l1_pred_expand:.4f}")
print(f"GT pitch var per-frame within first phoneme region: "
      f"{f0_norm_np[:gt_durs[0]].std():.4f}")
print(f"Pred pitch (constant within phoneme): "
      f"{pitch_pred_expanded_gt[0]:.4f}")

# Show pitch trajectory comparison for first 50 frames
print(f"\n  Frame-by-frame pitch (first 30 frames):")
print(f"  {'frame':>5} {'GT_pitch':>8} {'PredExp':>8} {'Diff':>8}")
for t in range(min(30, T_p)):
    print(f"  {t:>5} {f0_norm_np[t]:>8.3f} {pitch_pred_expanded_gt[t]:>8.3f} "
          f"{pitch_pred_expanded_gt[t]-f0_norm_np[t]:>+8.3f}")

# ── Step 5: Mel error heat map ──
print(f"\n{'='*70}")
print("4. MEL ERROR DISTRIBUTION (Config A vs C)")
print(f"{'='*70}")

mel_a_np = mel_a[0].numpy()
mel_c_np = mel_c[0].numpy()
mel_gt_np = mel_norm_gt[0].numpy()

err_a = np.abs(mel_a_np[:T_cmp] - mel_gt_np[:T_cmp]).mean(axis=1)  # per-frame error
err_c = np.abs(mel_c_np[:T_c_cmp] - mel_gt_np[:T_c_cmp]).mean(axis=1)

# Bin by phoneme
print(f"\n  Per-phoneme mel L1 (Config A vs C):")
print(f"  {'Phone':>6} {'GT_dur':>6} {'L1_A(GTvar)':>11} {'L1_C(Pred)':>11} {'Ratio':>6}")
print("  " + "-" * 45)
frame_a = 0
frame_c = 0
for i in range(min(15, len(gt_durs))):
    gd = gt_durs[i]
    pd_val = pred_durs[i] if i < len(pred_durs) else 0
    name = id2phone.get(phonemes[i], f"?{phonemes[i]}")

    # Config A error for this phoneme's frames
    if frame_a + gd <= len(err_a):
        a_err = err_a[frame_a:frame_a+gd].mean()
    else:
        a_err = -1
    # Config C error for this phoneme's frames
    if frame_c + pd_val <= len(err_c):
        c_err = err_c[frame_c:frame_c+pd_val].mean()
    else:
        c_err = -1

    ratio = c_err / a_err if a_err > 0.01 else 0
    print(f"  {name:>6} {gd:>6} {a_err:>11.3f} {c_err:>11.3f} {ratio:>6.1f}x")
    frame_a += gd
    frame_c += pd_val

# ── Step 6: Key insight - decoder input difference ──
print(f"\n{'='*70}")
print("5. DECODER INPUT ANALYSIS")
print(f"{'='*70}")

# Compare decoder input between config A and B
# Config A: enc_out expanded by GT dur + GT pitch/energy embed
# Config B: enc_out expanded by GT dur + PRED pitch/energy embed
# The ONLY difference is pitch/energy embedding values

dec_in_a = mi_a[0, :50]  # first 50 frames
dec_in_b = mi_b[0, :50]
dec_in_c = mi_c[0, :50]

diff_ab = (dec_in_a - dec_in_b).norm(dim=1)
diff_ac = (dec_in_a - dec_in_c).norm(dim=1)
diff_bc = (dec_in_b - dec_in_c).norm(dim=1)

T_in = min(diff_ac.size(0), diff_ab.size(0))
print(f"\n  Decoder input L2 distance (per frame, first {T_in} frames):")
print(f"    A vs B (pitch/energy effect):     mean={diff_ab[:T_in].mean():.3f}")
print(f"    A vs C (full pred effect):        mean={diff_ac[:T_in].mean():.3f}")
print(f"    B vs C (duration effect only):    mean={diff_bc[:T_in].mean():.3f}")
print(f"\n  Encoder output norm per frame: {enc_out[0].norm(dim=1).mean():.3f}")
print(f"  Pitch embed norm: {model.pitch_embed(f0_norm[:, :1].unsqueeze(-1)).norm():.3f}")
print(f"  Energy embed norm: {model.energy_embed(e_norm[:, :1].unsqueeze(-1)).norm():.3f}")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  GT-var mel L1 (A): {l1_a:.4f} — model fits well with teacher forcing")
print(f"  Pred pitch/energy effect (B-A): {l1_b-l1_a:+.4f}")
print(f"  Pred duration effect (C-B):     {l1_c-l1_b:+.4f}")
print(f"  Total train/inference gap (C-A): {l1_c-l1_a:+.4f}")
