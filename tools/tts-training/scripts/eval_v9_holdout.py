"""Evaluate v9 on holdout_20 using training forward path."""
import json, sys, math, numpy as np, torch, soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))
from nar_fastspeech2 import FastSpeech2
from nar_train import length_regulate_batch
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0'
FEAT_DIR = ROOT / 'data' / 'paddle_distill_features_v3'
STATS = json.load(open(ROOT / 'data' / 'train_300_norm_stats_v3.json'))
mel_mean = np.array(STATS['mel_mean'], dtype=np.float32)
mel_std = np.array(STATS['mel_std'], dtype=np.float32)

voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'), providers=['CPUExecutionProvider'])

ckpt = torch.load(str(ROOT / 'checkpoints' / 'fs2_final.pt'), map_location='cpu', weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
state_dict = {k: v for k, v in ckpt['model'].items() if not k.startswith('film_gen')}
model.load_state_dict(state_dict)
model.eval()
print("Loaded v9 (teacher dur), step=" + str(ckpt.get('step')))

holdout = [json.loads(l) for l in open(ROOT / 'data/holdout_20_manifest_v3.jsonl', encoding='utf-8')]
train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v3.jsonl', encoding='utf-8')]

asr = load_asr_model()

def infer_one(phone_ids):
    x = model.embedding(phone_ids) * math.sqrt(model.d_model)
    x = model.pos_enc(x)
    for layer in model.encoder_layers:
        x = layer(x)
    log_dur = model.duration_predictor(x)
    pitch_p = model.pitch_predictor(x)
    energy_p = model.energy_predictor(x)
    pred_dur = log_dur.exp().round().clamp(min=1).long()
    mel_input = length_regulate_batch(x, pred_dur)
    T = mel_input.size(1)
    pitch_exp = length_regulate_batch(pitch_p.unsqueeze(-1), pred_dur).squeeze(-1)
    energy_exp = length_regulate_batch(energy_p.unsqueeze(-1), pred_dur).squeeze(-1)
    p_emb = model.pitch_embed(pitch_exp[:, :T].unsqueeze(-1))
    e_emb = model.energy_embed(energy_exp[:, :T].unsqueeze(-1))
    mel_input = mel_input + p_emb + e_emb
    mel_input = model.pos_enc(mel_input)
    dec = mel_input
    for layer in model.decoder_layers:
        dec = layer(dec)
    return model.mel_linear(dec)


# --- Holdout ---
print("\n=== HOLDOUT (20 samples, unseen poems) ===")
OUTDIR = ROOT / 'output' / 'v9_holdout'
OUTDIR.mkdir(parents=True, exist_ok=True)
holdout_cers = []
for idx, rec in enumerate(holdout):
    pid = rec['poem_id']
    phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)
    text = rec['text']
    with torch.no_grad():
        mel_pred = infer_one(phone_ids)
    mel_raw = mel_pred[0].numpy() * mel_std + mel_mean
    audio = voc.run(None, {'logmel': mel_raw.astype(np.float32)})[0].flatten()
    wav = str(OUTDIR / (pid + '.wav'))
    sf.write(wav, audio, 24000)
    asr_text, _ = transcribe(asr, wav)
    cer, _, _, _, _ = cer_detail(text, asr_text)
    holdout_cers.append(cer)
    print("  [%2d/20] %s: CER=%.0f%% | %s" % (idx+1, pid, cer*100, asr_text[:40]))

hc = np.array(holdout_cers)
print("\nHoldout mean CER: %.1f%%" % (hc.mean()*100))
print("100%% CER: %d/20" % sum(1 for c in hc if c >= 0.99))
print("<30%% CER: %d/20" % sum(1 for c in hc if c < 0.30))

# --- Train (30 samples, for comparison) ---
print("\n=== TRAIN (30 samples, same as v9 eval) ===")
import random
random.seed(42)
sample = random.sample(train, 30)
train_cers = []
for idx, rec in enumerate(sample):
    pid = rec['poem_id']
    phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)
    text = rec['text']
    with torch.no_grad():
        mel_pred = infer_one(phone_ids)
    mel_raw = mel_pred[0].numpy() * mel_std + mel_mean
    audio = voc.run(None, {'logmel': mel_raw.astype(np.float32)})[0].flatten()
    wav = str(OUTDIR / ('train_' + pid + '.wav'))
    sf.write(wav, audio, 24000)
    asr_text, _ = transcribe(asr, wav)
    cer, _, _, _, _ = cer_detail(text, asr_text)
    train_cers.append(cer)

tc = np.array(train_cers)
print("\nTrain mean CER: %.1f%%" % (tc.mean()*100))

print("\n" + "="*60)
print("  Train CER:   %.1f%%" % (tc.mean()*100))
print("  Holdout CER: %.1f%%" % (hc.mean()*100))
print("  Gap: %.1fpp" % (hc.mean()*100 - tc.mean()*100))
print("="*60)
