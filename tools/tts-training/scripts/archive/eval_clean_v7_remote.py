"""Phase 2 eval: v7 clean classic FS2 — pred-all AND GT-all CER."""
import json, sys, random, numpy as np, torch, soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0'
FEAT_DIR = ROOT / 'data' / 'paddle_distill_features_v2'
STATS = json.load(open(ROOT / 'data' / 'train_300_norm_stats_v2.json'))
mel_mean = np.array(STATS['mel_mean'], dtype=np.float32)
mel_std = np.array(STATS['mel_std'], dtype=np.float32)

voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'), providers=['CPUExecutionProvider'])

ckpt = torch.load(str(ROOT / 'checkpoints' / 'fs2_final.pt'), map_location='cpu', weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
state_dict = {k: v for k, v in ckpt['model'].items() if not k.startswith('film_gen')}
model.load_state_dict(state_dict)
model.eval()
print("Loaded clean_v7, step=" + str(ckpt.get('step')))

train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v2.jsonl', encoding='utf-8')]
random.seed(42)
sample = random.sample(train, 30)
sample.sort(key=lambda r: r['mel_len'])

print('Loading ASR...')
asr = load_asr_model()


def eval_condition(name, use_gt_all=False):
    OUTDIR = ROOT / 'output' / ('v7_clean_24k_' + name)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, rec in enumerate(sample):
        pid = rec['poem_id']
        phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)
        text = rec['text']
        with torch.no_grad():
            if use_gt_all:
                npz = np.load(FEAT_DIR / f"{pid}.npz")
                dur_gt = torch.tensor([rec['durations']], dtype=torch.long)
                pitch_gt = torch.tensor([npz['f0']], dtype=torch.float)
                energy_gt = torch.tensor([npz['energy']], dtype=torch.float)
                mel_out, _, _, _ = model.forward(phone_ids, durations=dur_gt, pitches=pitch_gt, energies=energy_gt)
            else:
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
    print("\n=== v7 clean 24k %s CER ===" % name)
    print("Mean CER: %.1f%%" % (cers.mean()*100))
    print("100%% CER: %d/30" % sum(1 for c in cers if c >= 0.99))
    print("<15%% CER: %d/30" % sum(1 for c in cers if c < 0.15))
    return cers.mean()*100


pred_cer = eval_condition('pred', use_gt_all=False)
gt_cer = eval_condition('gt_all', use_gt_all=True)

print("\n" + "="*60)
print("=== v7 Clean Classic FS2 Summary ===")
print("  GT-all CER:  %.1f%%  (ceiling ~20.2%%, old=32.1%%)" % gt_cer)
print("  pred-all CER: %.1f%%  (old v2=77.5%%)" % pred_cer)
print("  Gap: %.1fpp" % (pred_cer - gt_cer))
print("="*60)
