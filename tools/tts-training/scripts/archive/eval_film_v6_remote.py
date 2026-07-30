import json, sys, random, numpy as np, torch, soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0'
STATS = json.load(open(ROOT / 'data' / 'train_300_norm_stats.json'))
mel_mean = np.array(STATS['mel_mean'], dtype=np.float32)
mel_std = np.array(STATS['mel_std'], dtype=np.float32)

voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'), providers=['CPUExecutionProvider'])

ckpt = torch.load(str(ROOT / 'checkpoints' / 'fs2_final.pt'), map_location='cpu', weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt['model'])
model.eval()
print("Loaded fs2_final (film_v6), step=" + str(ckpt.get('step')))

train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest.jsonl', encoding='utf-8')]
random.seed(42)
sample = random.sample(train, 30)
sample.sort(key=lambda r: r['mel_len'])

print('Loading ASR...')
asr = load_asr_model()

OUTDIR = ROOT / 'output' / 'v6_film_24k_pred'
OUTDIR.mkdir(parents=True, exist_ok=True)

results = []
for idx, rec in enumerate(sample):
    pid = rec['poem_id']
    phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)
    text = rec['text']
    with torch.no_grad():
        mel_out, _, _, _ = model.forward(phone_ids)
    mel_raw = mel_out[0].numpy() * mel_std + mel_mean
    audio = voc.run(None, {'logmel': mel_raw.astype(np.float32)})[0].flatten()
    wav = str(OUTDIR / (pid + '.wav'))
    sf.write(wav, audio, 24000)
    asr_text, _ = transcribe(asr, wav)
    cer, _, _, _, _ = cer_detail(text, asr_text)
    results.append(cer)
    if idx < 3 or idx % 10 == 0:
        print("  [%d/30] %s: CER=%.0f%% | %s" % (idx+1, pid, cer*100, text[:25]))

cers = np.array(results)
print("\n=== film_v6 step24k pred-all CER ===")
print("Mean CER: %.1f%%" % (cers.mean()*100))
print("100%% CER: %d/30" % sum(1 for c in cers if c >= 0.99))
print("<15%% CER: %d/30" % sum(1 for c in cers if c < 0.15))
print("\nFor reference:")
print("  Full E2E v2 pred-all CER = 77.5%")
print("  Phoneme-level v3 pred-all CER = 84.2%")
print("  Alternating v4 pred-all CER = 95.3%")
print("  Multiplicative gate v5 pred-all CER = 83.2%")
