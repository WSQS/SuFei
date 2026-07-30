"""Extract teacher durations for all 320 samples, rebuild manifest + stats."""
import json, sys, numpy as np
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))

FS2_DUR_ONNX = ROOT / 'models' / 'paddlespeech_onnx' / 'fastspeech2_csmsc_with_dur.onnx'
FEAT_DIR = ROOT / 'data' / 'paddle_distill_features_v2'
OUT_FEAT = ROOT / 'data' / 'paddle_distill_features_v3'

sess = ort.InferenceSession(str(FS2_DUR_ONNX), providers=['CPUExecutionProvider'])

OUT_FEAT.mkdir(parents=True, exist_ok=True)

for manifest_name in ['train_300_manifest_v2.jsonl', 'holdout_20_manifest_v2.jsonl']:
    manifest_path = ROOT / 'data' / manifest_name
    records = [json.loads(l) for l in open(manifest_path, encoding='utf-8')]

    out_name = manifest_name.replace('_v2', '_v3')
    out_path = ROOT / 'data' / out_name

    new_records = []
    for idx, rec in enumerate(records):
        pid = rec['poem_id']
        phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)

        outputs = sess.run(None, {"text": phone_ids})
        mel = outputs[0]
        if mel.ndim == 3: mel = mel[0]
        teacher_dur = outputs[1][0].astype(int)  # [n_phonemes]

        assert teacher_dur.sum() == mel.shape[0], \
            "%s: dur_sum=%d != mel_frames=%d" % (pid, teacher_dur.sum(), mel.shape[0])

        # Load existing features (mel/f0/energy from v2)
        npz = np.load(FEAT_DIR / f"{pid}.npz")

        # Save with teacher duration-compatible features
        # mel stays the same (teacher_mel), f0/energy stay same
        np.savez_compressed(
            str(OUT_FEAT / f"{pid}.npz"),
            mel=npz['mel'],
            f0=npz['f0'],
            energy=npz['energy'],
        )

        new_rec = dict(rec)
        new_rec['durations'] = teacher_dur.tolist()
        new_rec['mel_len'] = int(teacher_dur.sum())
        new_rec['mel_path'] = f"data/paddle_distill_features_v3/{pid}.npz"
        new_records.append(new_rec)

        if (idx + 1) % 50 == 0 or idx == 0:
            print("  [%d/%d] %s: %d phones, mel_len=%d" % (
                idx+1, len(records), pid, len(phone_ids), teacher_dur.sum()))

    with open(out_path, 'w', encoding='utf-8') as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print("Saved: %s (%d records)" % (out_path, len(new_records)))

# Compute new norm stats (only durations change, mel/f0/energy same)
print("\nComputing norm stats (v3)...")
import numpy as np

train_records = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v3.jsonl', encoding='utf-8')]

all_mel = []
all_f0_voiced = []
all_energy = []
all_dur = []

for r in train_records:
    npz = np.load(OUT_FEAT / f"{r['poem_id']}.npz")
    all_mel.append(npz['mel'].astype(np.float32))
    voiced = npz['f0'][npz['f0'] > 0]
    if len(voiced) > 0:
        all_f0_voiced.append(np.log(voiced))
    all_energy.append(npz['energy'].astype(np.float32))
    all_dur.extend(r['durations'])

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

stats_path = ROOT / 'data' / 'train_300_norm_stats_v3.json'
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)

print("Saved: %s" % stats_path)

# Duration stats
all_dur = np.array(all_dur)
print("\nDuration stats (teacher):")
print("  mean=%.1f, median=%d, dur<=2: %.1f%%, dur<=5: %.1f%%" % (
    all_dur.mean(), np.median(all_dur),
    (all_dur <= 2).sum()/len(all_dur)*100,
    (all_dur <= 5).sum()/len(all_dur)*100))
print("  max=%d (vs MFA had 152)" % all_dur.max())

print("\nDone!")
