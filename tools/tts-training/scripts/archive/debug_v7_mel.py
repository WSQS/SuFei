"""Debug: check mel scale mismatch between teacher_mel and what eval feeds to HiFiGAN."""
import json, sys, numpy as np, torch, soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))
from nar_fastspeech2 import FastSpeech2

HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0'
FS2_DIR = ROOT / 'models' / 'paddlespeech_onnx' / 'fastspeech2_csmsc_onnx_0.2.0'
FEAT_DIR = ROOT / 'data' / 'paddle_distill_features_v2'
STATS = json.load(open(ROOT / 'data' / 'train_300_norm_stats_v2.json'))

mel_mean = np.array(STATS['mel_mean'], dtype=np.float32)
mel_std = np.array(STATS['mel_std'], dtype=np.float32)

# Load one sample
train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v2.jsonl', encoding='utf-8')]
rec = train[0]
pid = rec['poem_id']
print("Sample:", pid)

# 1. teacher_mel (what HiFiGAN expects)
fs2 = ort.InferenceSession(str(FS2_DIR / 'fastspeech2_csmsc.onnx'), providers=['CPUExecutionProvider'])
phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)
teacher_mel = fs2.run(None, {"text": phone_ids})[0]
if teacher_mel.ndim == 3:
    teacher_mel = teacher_mel[0]
print("teacher_mel shape:", teacher_mel.shape)
print("teacher_mel range: [%.2f, %.2f], mean=%.2f" % (teacher_mel.min(), teacher_mel.max(), teacher_mel.mean()))

# 2. teacher_mel stored in npz (should be same)
npz = np.load(FEAT_DIR / f"{pid}.npz")
stored_mel = npz['mel']
print("\nstored mel shape:", stored_mel.shape)
print("stored mel range: [%.2f, %.2f], mean=%.2f" % (stored_mel.min(), stored_mel.max(), stored_mel.mean()))
print("stored == teacher:", np.allclose(stored_mel, teacher_mel, atol=1e-4))

# 3. What eval script does: model output (normalized) * std + mean
ckpt = torch.load(str(ROOT / 'checkpoints' / 'fs2_final.pt'), map_location='cpu', weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt['model'])
model.eval()

dur_gt = torch.tensor([rec['durations']], dtype=torch.long)
pitch_gt = torch.tensor([npz['f0']], dtype=torch.float)
energy_gt = torch.tensor([npz['energy']], dtype=torch.float)
phone_t = torch.tensor([rec['phoneme_ids']], dtype=torch.long)

with torch.no_grad():
    mel_out, _, _, _ = model.forward(phone_t, durations=dur_gt, pitches=pitch_gt, energies=energy_gt)

print("\nmodel output (normalized) range: [%.4f, %.4f], mean=%.4f" % (
    mel_out[0].min().item(), mel_out[0].max().item(), mel_out[0].mean().item()))

mel_denorm = mel_out[0].numpy() * mel_std + mel_mean
print("denorm mel range: [%.2f, %.2f], mean=%.2f" % (
    mel_denorm.min(), mel_denorm.max(), mel_denorm.mean()))

# Compare with teacher_mel
T = min(mel_denorm.shape[0], teacher_mel.shape[0])
l1 = np.abs(mel_denorm[:T] - teacher_mel[:T]).mean()
print("\nL1 (denorm vs teacher): %.4f" % l1)
print("L1 (normalized vs (teacher-mean)/std): %.4f" % np.abs(((teacher_mel[:T] - mel_mean) / mel_std - mel_out[0].numpy()[:T])).mean())

# 4. Check: does HiFiGAN produce audio from teacher_mel vs denorm mel?
voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'), providers=['CPUExecutionProvider'])

audio_teacher = voc.run(None, {'logmel': teacher_mel[:T].astype(np.float32)})[0].flatten()
audio_model = voc.run(None, {'logmel': mel_denorm[:T].astype(np.float32)})[0].flatten()

print("\nAudio from teacher_mel: range=[%.4f, %.4f], rms=%.4f" % (
    audio_teacher.min(), audio_teacher.max(), np.sqrt(np.mean(audio_teacher**2))))
print("Audio from model denorm: range=[%.4f, %.4f], rms=%.4f" % (
    audio_model.min(), audio_model.max(), np.sqrt(np.mean(audio_model**2))))

# Save both
sf.write(str(ROOT / 'output' / 'debug_teacher.wav'), audio_teacher, 24000)
sf.write(str(ROOT / 'output' / 'debug_model.wav'), audio_model, 24000)
print("\nSaved debug_teacher.wav and debug_model.wav")
