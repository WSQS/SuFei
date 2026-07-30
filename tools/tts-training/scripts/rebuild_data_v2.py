"""Phase 1: Rebuild training data with clean teacher_mel + fixed F0 hop.

Changes vs old pipeline:
  1. mel = teacher FS2 output (NOT re-extracted from audio)
  2. F0 extracted with correct pyworld hop (120 samples, not 256)
  3. Energy extracted from teacher_mel (sum across bins)
  4. Durations re-adjusted to match teacher_mel length
"""
import json, sys, os
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))

FS2_DIR = ROOT / 'models' / 'paddlespeech_onnx' / 'fastspeech2_csmsc_onnx_0.2.0'
HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0'

SR = 24000
HOP_LENGTH = 300
F0_HOP = 120  # pyworld dio default: 5ms * 24000 / 1000 = 120

# Load existing manifests
train_manifest = ROOT / 'data' / 'train_300_manifest.jsonl'
val_manifest = ROOT / 'data' / 'holdout_20_manifest.jsonl'

train_records = [json.loads(l) for l in open(train_manifest, encoding='utf-8')]
val_records = [json.loads(l) for l in open(val_manifest, encoding='utf-8')]
all_records = train_records + val_records

print("Records: %d train + %d val = %d total" % (len(train_records), len(val_records), len(all_records)))

# Load models
fs2 = ort.InferenceSession(str(FS2_DIR / 'fastspeech2_csmsc.onnx'), providers=['CPUExecutionProvider'])
voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'), providers=['CPUExecutionProvider'])

# Output dirs
OUT_FEAT = ROOT / 'data' / 'paddle_distill_features_v2'
OUT_FEAT.mkdir(parents=True, exist_ok=True)

new_records = []
for idx, rec in enumerate(all_records):
    pid = rec['poem_id']
    phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)

    # 1. Get teacher mel (the SAME mel HiFiGAN was trained on)
    teacher_mel = fs2.run(None, {"text": phone_ids})[0]
    if teacher_mel.ndim == 3:
        teacher_mel = teacher_mel[0]
    teacher_mel = teacher_mel.astype(np.float32)  # [T, 80]

    # 2. Vocoded audio for F0 extraction
    audio = voc.run(None, {'logmel': teacher_mel.astype(np.float32)})[0].flatten()

    # 3. Extract F0 with CORRECT hop (120, not 256)
    import pyworld as pyworld
    audio_d = audio.astype(np.float64)
    _f0, t = pyworld.dio(audio_d, SR, f0_floor=80, f0_ceil=400)
    f0_pw = pyworld.stonemask(audio_d, _f0, t, SR)

    mel_len = teacher_mel.shape[0]
    f0_fixed = np.zeros(mel_len, dtype=np.float32)
    for i in range(mel_len):
        start = i * HOP_LENGTH
        end = min((i + 1) * HOP_LENGTH, len(audio))
        # CORRECT: use F0_HOP=120, not 256
        frame_f0 = f0_pw[start // F0_HOP : end // F0_HOP]
        voiced = frame_f0[frame_f0 > 0]
        if len(voiced) > 0:
            f0_fixed[i] = np.mean(voiced)

    # 4. Energy from teacher_mel (sum across bins, same as old pipeline)
    energy = np.sum(teacher_mel, axis=1).astype(np.float32)

    # 5. Save features
    np.savez_compressed(
        str(OUT_FEAT / f"{pid}.npz"),
        mel=teacher_mel,
        f0=f0_fixed,
        energy=energy,
    )

    # 6. Adjust durations to match new mel_len
    old_durations = rec['durations']
    old_mel_len = rec['mel_len']
    dur_sum = sum(old_durations)

    if dur_sum != mel_len:
        diff = mel_len - dur_sum
        max_idx = max(range(len(old_durations)), key=lambda x: old_durations[x])
        old_durations = list(old_durations)
        old_durations[max_idx] = max(old_durations[max_idx] + diff, 1)

    assert sum(old_durations) == mel_len, "%s: dur_sum=%d != mel_len=%d" % (pid, sum(old_durations), mel_len)

    new_rec = dict(rec)
    new_rec['mel_len'] = mel_len
    new_rec['durations'] = old_durations
    new_rec['mel_path'] = f"data/paddle_distill_features_v2/{pid}.npz"
    new_records.append(new_rec)

    if (idx + 1) % 50 == 0 or idx == 0:
        print("  [%d/%d] %s: mel_len=%d, f0_voiced=%d/%d" % (
            idx+1, len(all_records), pid, mel_len,
            np.sum(f0_fixed > 0), mel_len
        ))

# Split back into train/val
new_train = [r for r in new_records if r['poem_id'] in {r2['poem_id'] for r2 in train_records}]
new_val = [r for r in new_records if r['poem_id'] in {r2['poem_id'] for r2 in val_records}]

train_out = ROOT / 'data' / 'train_300_manifest_v2.jsonl'
val_out = ROOT / 'data' / 'holdout_20_manifest_v2.jsonl'

with open(train_out, 'w', encoding='utf-8') as f:
    for r in new_train:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

with open(val_out, 'w', encoding='utf-8') as f:
    for r in new_val:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print("\nSaved manifests:")
print("  Train: %s (%d records)" % (train_out, len(new_train)))
print("  Val: %s (%d records)" % (val_out, len(new_val)))
print("  Features: %s" % OUT_FEAT)

# 7. Compute new norm stats
print("\nComputing norm stats...")
all_mel = []
all_f0_voiced = []
all_energy = []

for r in new_train:
    npz = np.load(OUT_FEAT / f"{r['poem_id']}.npz")
    all_mel.append(npz['mel'].astype(np.float32))
    voiced = npz['f0'][npz['f0'] > 0]
    if len(voiced) > 0:
        all_f0_voiced.append(np.log(voiced))
    all_energy.append(npz['energy'].astype(np.float32))

mel_cat = np.concatenate(all_mel, axis=0)
mel_mean = mel_cat.mean(axis=0)
mel_std = mel_cat.std(axis=0) + 1e-8

f0_cat = np.concatenate(all_f0_voiced)
f0_mean = float(f0_cat.mean())
f0_std = float(f0_cat.std()) + 1e-8

energy_cat = np.concatenate(all_energy)
energy_mean = float(energy_cat.mean())
energy_std = float(energy_cat.std()) + 1e-8

stats = {
    "mel_mean": mel_mean.tolist(),
    "mel_std": mel_std.tolist(),
    "f0_mean": f0_mean,
    "f0_std": f0_std,
    "energy_mean": energy_mean,
    "energy_std": energy_std,
}

stats_path = ROOT / 'data' / 'train_300_norm_stats_v2.json'
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)

print("Saved: %s" % stats_path)
print("mel_mean range: [%.2f, %.2f]" % (mel_mean.min(), mel_mean.max()))
print("mel_std range:  [%.4f, %.4f]" % (mel_std.min(), mel_std.max()))
print("f0_mean=%.2f f0_std=%.2f" % (f0_mean, f0_std))
print("energy_mean=%.2f energy_std=%.2f" % (energy_mean, energy_std))

print("\nFor reference (old stats):")
old_stats = json.load(open(ROOT / 'data' / 'train_300_norm_stats.json'))
print("  old mel_mean range: [%.2f, %.2f]" % (min(old_stats['mel_mean']), max(old_stats['mel_mean'])))
print("  old mel_std range:  [%.4f, %.4f]" % (min(old_stats['mel_std']), max(old_stats['mel_std'])))
print("  old f0_mean=%.2f f0_std=%.2f" % (old_stats['f0_mean'], old_stats['f0_std']))
print("  old energy_mean=%.2f energy_std=%.2f" % (old_stats['energy_mean'], old_stats['energy_std']))
print("\nDone!")
