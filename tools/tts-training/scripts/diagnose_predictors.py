"""Compare duration predictor patterns: train vs holdout.

Key question: does the model memorize per-poem durations, or does it
learn generalizable phoneme-level duration rules?
"""
import json, sys, math, numpy as np, torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nar_fastspeech2 import FastSpeech2

ROOT = Path(__file__).resolve().parents[1]
ckpt = torch.load(str(ROOT / "checkpoints" / "D300fixE2E_step24000_slim.pt"),
                   map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()

phone_map = {}
with open(ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0" / "phone_id_map.txt",
          encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            phone_map[parts[0]] = int(parts[1])
id2phone = {v: k for k, v in phone_map.items()}

train = [json.loads(l) for l in open(ROOT / "data" / "train_300_manifest.jsonl", encoding="utf-8")]
holdout = [json.loads(l) for l in open(ROOT / "data" / "holdout_20_manifest.jsonl", encoding="utf-8")]


def get_pred_durations(model, rec):
    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        log_dur = model.duration_predictor(x)
    return np.maximum(np.round(np.exp(log_dur[0].numpy())), 0).astype(int)


def get_pred_pitch(model, rec):
    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        pitch = model.pitch_predictor(x)
    return pitch[0].numpy()


# Collect per-phoneme statistics: what does the model predict for each phoneme ID?
phoneme_pred_durs = defaultdict(list)
phoneme_gt_durs = defaultdict(list)

for rec in train:
    pred = get_pred_durations(model, rec)
    gt = np.array(rec["durations"])
    for pid, pd_val, gt_val in zip(rec["phoneme_ids"], pred, gt):
        phoneme_pred_durs[pid].append(pd_val)
        phoneme_gt_durs[pid].append(gt_val)

print("=" * 70)
print("Per-phoneme duration statistics (train_300)")
print("=" * 70)
print(f"{'Phone':>6} {'n':>4} {'GT_mean':>7} {'Pred_mean':>9} {'Pred_std':>8} {'Ratio':>6}")
print("-" * 45)

# Show top 20 most common phonemes
common = sorted(phoneme_gt_durs.keys(), key=lambda k: len(phoneme_gt_durs[k]), reverse=True)[:20]
for pid in common:
    gt_vals = np.array(phoneme_gt_durs[pid])
    pred_vals = np.array(phoneme_pred_durs[pid])
    phone_name = id2phone.get(pid, f"?{pid}")
    ratio = pred_vals.mean() / max(gt_vals.mean(), 0.1)
    print(f"{phone_name:>6} {len(gt_vals):>4} {gt_vals.mean():>7.1f} "
          f"{pred_vals.mean():>9.1f} {pred_vals.std():>8.1f} {ratio:>6.2f}")

# Key metric: correlation between predicted and GT per-phoneme
all_gt = []
all_pred = []
for rec in train[:50]:
    pred = get_pred_durations(model, rec)
    gt = np.array(rec["durations"])
    all_gt.extend(gt.tolist())
    all_pred.extend(pred.tolist())
all_gt = np.array(all_gt)
all_pred = np.array(all_pred)
corr = np.corrcoef(all_gt, all_pred)[0, 1]
print(f"\nTrain per-phoneme correlation: {corr:.3f}")

# Same for holdout
all_gt_h = []
all_pred_h = []
for rec in holdout:
    pred = get_pred_durations(model, rec)
    gt = np.array(rec["durations"])
    all_gt_h.extend(gt.tolist())
    all_pred_h.extend(pred.tolist())
all_gt_h = np.array(all_gt_h)
all_pred_h = np.array(all_pred_h)
corr_h = np.corrcoef(all_gt_h, all_pred_h)[0, 1]
print(f"Holdout per-phoneme correlation: {corr_h:.3f}")

# Pitch analysis
print(f"\n{'='*70}")
print("Pitch predictor analysis")
print("=" * 70)

train_pitch_stds = []
for rec in train[:50]:
    pitch = get_pred_pitch(model, rec)
    train_pitch_stds.append(pitch.std())

holdout_pitch_stds = []
for rec in holdout:
    pitch = get_pred_pitch(model, rec)
    holdout_pitch_stds.append(pitch.std())

print(f"Train pitch std: mean={np.mean(train_pitch_stds):.3f} "
      f"min={np.min(train_pitch_stds):.3f} max={np.max(train_pitch_stds):.3f}")
print(f"Holdout pitch std: mean={np.mean(holdout_pitch_stds):.3f} "
      f"min={np.min(holdout_pitch_stds):.3f} max={np.max(holdout_pitch_stds):.3f}")

# Check if model outputs same duration for same phoneme regardless of context
print(f"\n{'='*70}")
print("Context sensitivity: does the model output different durations")
print("for the same phoneme in different poems?")
print("=" * 70)

for pid in common[:5]:
    preds = phoneme_pred_durs[pid]
    if len(preds) > 5:
        phone_name = id2phone.get(pid, f"?{pid}")
        print(f"  {phone_name} (id={pid}): n={len(preds)} "
              f"values={sorted(set(preds))[:10]} "
              f"unique={len(set(preds))}/{len(preds)}")
