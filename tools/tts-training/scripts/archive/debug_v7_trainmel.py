"""Debug: check what mel the model actually sees during training vs what we evaluate with."""
import json, sys, numpy as np, torch
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))
from nar_fastspeech2 import FastSpeech2

FEAT_DIR = ROOT / 'data' / 'paddle_distill_features_v2'
STATS = json.load(open(ROOT / 'data' / 'train_300_norm_stats_v2.json'))
mel_mean = np.array(STATS['mel_mean'], dtype=np.float32)
mel_std = np.array(STATS['mel_std'], dtype=np.float32)

ckpt = torch.load(str(ROOT / 'checkpoints' / 'fs2_final.pt'), map_location='cpu', weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt['model'])
model.eval()

train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v2.jsonl', encoding='utf-8')]

# Take 3 training samples and check GT mel L1
for rec in train[:3]:
    pid = rec['poem_id']
    npz = np.load(FEAT_DIR / f"{pid}.npz")
    mel_gt = npz['mel']  # teacher_mel, unnormalized

    # Normalize
    mel_norm = (mel_gt - mel_mean) / mel_std

    phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)
    dur_gt = torch.tensor([rec['durations']], dtype=torch.long)

    # GT pitch/energy
    f0 = npz['f0']
    energy = npz['energy']
    # Normalize f0
    f0_norm = np.where(f0 > 0, (np.log(f0.clip(min=1)) - STATS['f0_mean']) / STATS['f0_std'], np.zeros_like(f0))
    energy_norm = (energy - STATS['energy_mean']) / STATS['energy_std']

    pitch_gt = torch.tensor([f0_norm], dtype=torch.float)
    energy_gt = torch.tensor([energy_norm], dtype=torch.float)

    with torch.no_grad():
        mel_out, _, _, _ = model.forward(phone_ids, durations=dur_gt, pitches=pitch_gt, energies=energy_gt)

    mel_out_np = mel_out[0].numpy()
    T = min(mel_out_np.shape[0], mel_norm.shape[0])

    l1_norm = np.abs(mel_out_np[:T] - mel_norm[:T]).mean()
    l1_raw = np.abs((mel_out_np[:T] * mel_std + mel_mean) - mel_gt[:T]).mean()

    print("%s: mel_len=%d, L1(norm)=%.4f, L1(raw)=%.4f" % (pid, T, l1_norm, l1_raw))
    print("  mel_out range: [%.2f, %.2f], mel_norm range: [%.2f, %.2f]" % (
        mel_out_np.min(), mel_out_np.max(), mel_norm.min(), mel_norm.max()))

# Now try: feed teacher_mel directly to HiFiGAN (bypass model) — should give ~20% CER
print("\n--- Sanity: teacher_mel -> HiFiGAN -> ASR ---")
import onnxruntime as ort
import soundfile as sf
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

FS2_DIR = ROOT / 'models' / 'paddlespeech_onnx' / 'fastspeech2_csmsc_onnx_0.2.0'
HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0'

fs2 = ort.InferenceSession(str(FS2_DIR / 'fastspeech2_csmsc.onnx'), providers=['CPUExecutionProvider'])
voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'), providers=['CPUExecutionProvider'])
asr = load_asr_model()

OUTDIR = ROOT / 'output' / 'debug_sanity'
OUTDIR.mkdir(parents=True, exist_ok=True)

cers = []
for rec in train[:5]:
    pid = rec['poem_id']
    phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)
    text = rec['text']
    teacher_mel = fs2.run(None, {"text": phone_ids})[0]
    if teacher_mel.ndim == 3:
        teacher_mel = teacher_mel[0]
    audio = voc.run(None, {'logmel': teacher_mel.astype(np.float32)})[0].flatten()
    wav = str(OUTDIR / f"{pid}_teacher.wav")
    sf.write(wav, audio, 24000)
    asr_text, _ = transcribe(asr, wav)
    cer, _, _, _, _ = cer_detail(text, asr_text)
    cers.append(cer)
    print("  %s: CER=%.0f%% | %s" % (pid, cer*100, asr_text[:30]))

print("Mean teacher CER: %.1f%%" % (np.mean(cers)*100))

# Now model GT-all for same samples
print("\n--- Model GT-all -> HiFiGAN -> ASR ---")
cers2 = []
for rec in train[:5]:
    pid = rec['poem_id']
    text = rec['text']
    npz = np.load(FEAT_DIR / f"{pid}.npz")
    mel_gt = npz['mel']
    f0 = npz['f0']
    energy = npz['energy']
    f0_norm = np.where(f0 > 0, (np.log(f0.clip(min=1)) - STATS['f0_mean']) / STATS['f0_std'], np.zeros_like(f0))
    energy_norm = (energy - STATS['energy_mean']) / STATS['energy_std']

    phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)
    dur_gt = torch.tensor([rec['durations']], dtype=torch.long)
    pitch_gt = torch.tensor([f0_norm], dtype=torch.float)
    energy_gt = torch.tensor([energy_norm], dtype=torch.float)

    with torch.no_grad():
        mel_out, _, _, _ = model.forward(phone_ids, durations=dur_gt, pitches=pitch_gt, energies=energy_gt)

    mel_raw = mel_out[0].numpy() * mel_std + mel_mean
    audio = voc.run(None, {'logmel': mel_raw.astype(np.float32)})[0].flatten()
    wav = str(OUTDIR / f"{pid}_model_gt.wav")
    sf.write(wav, audio, 24000)
    asr_text, _ = transcribe(asr, wav)
    cer, _, _, _, _ = cer_detail(text, asr_text)
    cers2.append(cer)
    print("  %s: CER=%.0f%% | %s" % (pid, cer*100, asr_text[:30]))

print("Mean model GT-all CER: %.1f%%" % (np.mean(cers2)*100))
