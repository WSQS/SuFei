"""Debug: check if MFA durations align with teacher_mel frames."""
import json, sys, numpy as np, torch
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))

FS2_DIR = ROOT / 'models' / 'paddlespeech_onnx' / 'fastspeech2_csmsc_onnx_0.2.0'
HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0'
FEAT_DIR = ROOT / 'data' / 'paddle_distill_features_v2'
OLD_FEAT_DIR = ROOT / 'data' / 'paddle_distill_features'

fs2 = ort.InferenceSession(str(FS2_DIR / 'fastspeech2_csmsc.onnx'), providers=['CPUExecutionProvider'])
voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'), providers=['CPUExecutionProvider'])

train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v2.jsonl', encoding='utf-8')]

# Check 5 samples
for rec in train[:5]:
    pid = rec['poem_id']
    phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)

    # teacher_mel
    teacher_mel = fs2.run(None, {"text": phone_ids})[0]
    if teacher_mel.ndim == 3:
        teacher_mel = teacher_mel[0]

    # old mel (re-extracted from audio)
    old_npz = np.load(OLD_FEAT_DIR / f"{pid}.npz")
    old_mel = old_npz['mel']

    # v2 mel (should = teacher_mel)
    v2_npz = np.load(FEAT_DIR / f"{pid}.npz")
    v2_mel = v2_npz['mel']

    # Durations
    dur = rec['durations']

    print("\n%s: %d phonemes, %d durations" % (pid, len(rec['phoneme_ids']), len(dur)))
    print("  teacher_mel: %d frames" % teacher_mel.shape[0])
    print("  old mel:     %d frames" % old_mel.shape[0])
    print("  v2 mel:      %d frames" % v2_mel.shape[0])
    print("  sum(dur):    %d" % sum(dur))
    print("  v2 == teacher: %s" % np.allclose(v2_mel, teacher_mel, atol=1e-4))

    # Generate audio from teacher_mel and old_mel, compare ASR
    import soundfile as sf
    audio_teacher = voc.run(None, {'logmel': teacher_mel.astype(np.float32)})[0].flatten()
    audio_old = voc.run(None, {'logmel': old_mel.astype(np.float32)})[0].flatten()

    # Check where old mel and teacher mel differ
    T = min(old_mel.shape[0], teacher_mel.shape[0])
    l1 = np.abs(old_mel[:T] - teacher_mel[:T]).mean()
    print("  L1(old vs teacher) for first %d frames: %.4f" % (T, l1))
    print("  old mel range: [%.2f, %.2f], teacher range: [%.2f, %.2f]" % (
        old_mel.min(), old_mel.max(), teacher_mel.min(), teacher_mel.max()))

    # Check frame-level alignment: look at energy contour
    old_energy = np.sum(old_mel, axis=1)
    teacher_energy = np.sum(teacher_mel, axis=1)
    # Cross-correlation at lag 0
    T_min = min(len(old_energy), len(teacher_energy))
    corr = np.corrcoef(old_energy[:T_min], teacher_energy[:T_min])[0, 1]
    print("  Energy correlation old vs teacher: %.3f" % corr)

# Now check if PaddleSpeech FS2 ONNX has multiple outputs (maybe durations?)
print("\n--- FS2 ONNX outputs ---")
for o in fs2.get_outputs():
    print("  %s: %s" % (o.name, o.type))
