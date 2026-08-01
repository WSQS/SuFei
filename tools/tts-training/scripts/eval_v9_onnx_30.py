"""Compare ONNX vs torch CER on the same 30 train samples."""
import json, sys, numpy as np, torch, soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')

HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0'
ONNX_MODEL = ROOT / 'models' / 'sufei_fs2_onnx' / 'fastspeech2_sufei.onnx'
stats = json.load(open(ROOT / 'data' / 'train_300_norm_stats_v3.json'))
mel_mean = np.array(stats['mel_mean'], dtype=np.float32)
mel_std = np.array(stats['mel_std'], dtype=np.float32)

sess = ort.InferenceSession(str(ONNX_MODEL), providers=['CPUExecutionProvider'])
voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'), providers=['CPUExecutionProvider'])

sys.path.insert(0, str(ROOT / 'scripts'))
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

asr = load_asr_model()

train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v3.jsonl', encoding='utf-8')]
import random
random.seed(42)
sample = random.sample(train, 30)

OUTDIR = ROOT / 'output' / 'v9_onnx_30'
OUTDIR.mkdir(parents=True, exist_ok=True)

cers = []
for idx, rec in enumerate(sample):
    pid = rec['poem_id']
    phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)
    text = rec['text']
    mel_norm = sess.run(None, {"text": phone_ids})[0]
    mel_raw = mel_norm * mel_std + mel_mean
    audio = voc.run(None, {'logmel': mel_raw.astype(np.float32)})[0].flatten()
    wav = str(OUTDIR / (pid + '.wav'))
    sf.write(wav, audio, 24000)
    asr_text, _ = transcribe(asr, wav)
    cer, _, _, _, _ = cer_detail(text, asr_text)
    cers.append(cer)
    print("[%2d/30] %s: CER=%.0f%% | ref=%s | asr=%s" % (idx+1, pid, cer*100, text[:25], asr_text[:25]))

c = np.array(cers)
print("\nONNX E2E (30 samples): mean CER = %.1f%%" % (c.mean()*100))
print("torch was 24.0% on same 30 samples")
print("Gap: %.1fpp" % (c.mean()*100 - 24.0))
print("<30%%: %d/30, <20%%: %d/30, <15%%: %d/30" % (
    sum(1 for x in c if x < 0.30),
    sum(1 for x in c if x < 0.20),
    sum(1 for x in c if x < 0.15),
))
