"""Phase 0 verification: teacher_mel -> HiFiGAN -> ASR should give ~6.8% CER."""
import json, sys, random, numpy as np, soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

FS2_DIR = ROOT / 'models' / 'paddlespeech_onnx' / 'fastspeech2_csmsc_onnx_0.2.0'
HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0'

fs2 = ort.InferenceSession(str(FS2_DIR / 'fastspeech2_csmsc.onnx'), providers=['CPUExecutionProvider'])
voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'), providers=['CPUExecutionProvider'])

train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest.jsonl', encoding='utf-8')]
random.seed(42)
sample = random.sample(train, 30)

print('Loading ASR...')
asr = load_asr_model()

OUTDIR = ROOT / 'output' / 'p0_teacher_mel_check'
OUTDIR.mkdir(parents=True, exist_ok=True)

results = []
for idx, rec in enumerate(sample):
    pid = rec['poem_id']
    phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)
    text = rec['text']

    # teacher mel (the SAME mel HiFiGAN was trained on)
    teacher_mel = fs2.run(None, {"text": phone_ids})[0]
    if teacher_mel.ndim == 3:
        teacher_mel = teacher_mel[0]

    # teacher_mel -> HiFiGAN -> audio
    audio = voc.run(None, {'logmel': teacher_mel.astype(np.float32)})[0].flatten()
    wav = str(OUTDIR / (pid + '.wav'))
    sf.write(wav, audio, 24000)
    asr_text, _ = transcribe(asr, wav)
    cer, _, _, _, _ = cer_detail(text, asr_text)
    results.append(cer)
    if idx < 5 or idx % 10 == 0:
        print("  [%d/30] %s: CER=%.0f%% | %s" % (idx+1, pid, cer*100, text[:25]))

cers = np.array(results)
print("\n=== Phase 0: teacher_mel -> HiFiGAN -> ASR ===")
print("Mean CER: %.1f%%" % (cers.mean()*100))
print("100%% CER: %d/30" % sum(1 for c in cers if c >= 0.99))
print("<15%% CER: %d/30" % sum(1 for c in cers if c < 0.15))
print("\nFor reference:")
print("  P2 (re-extracted mel -> HiFiGAN -> ASR) = 31.8%")
print("  PaddleSpeech production CER = 6.8%")
