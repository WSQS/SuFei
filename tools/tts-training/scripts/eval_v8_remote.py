"""Phase 2 eval v8: full_e2e on clean data — pred-all CER."""
import json, sys, math, random, numpy as np, torch, soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))
from nar_fastspeech2 import FastSpeech2
from nar_train import length_regulate_batch
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
print("Loaded clean_v8 (full_e2e), step=" + str(ckpt.get('step')))

train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v2.jsonl', encoding='utf-8')]
random.seed(42)
sample = random.sample(train, 30)
sample.sort(key=lambda r: r['mel_len'])

asr = load_asr_model()

def eval_pred_all():
    OUTDIR = ROOT / 'output' / 'v8_fulle2e_24k_pred'
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, rec in enumerate(sample):
        pid = rec['poem_id']
        phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)
        text = rec['text']
        with torch.no_grad():
            x = model.embedding(phone_ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x)
            log_dur = model.duration_predictor(x)
            pitch_p = model.pitch_predictor(x)
            energy_p = model.energy_predictor(x)
            pred_dur = log_dur.exp().round().clamp(min=1).long()
            mel_input = length_regulate_batch(x, pred_dur)
            T_pred = mel_input.size(1)
            pitch_exp = length_regulate_batch(pitch_p.unsqueeze(-1), pred_dur).squeeze(-1)
            energy_exp = length_regulate_batch(energy_p.unsqueeze(-1), pred_dur).squeeze(-1)
            p_emb = model.pitch_embed(pitch_exp[:, :T_pred].unsqueeze(-1))
            e_emb = model.energy_embed(energy_exp[:, :T_pred].unsqueeze(-1))
            mel_input = mel_input + p_emb + e_emb
            mel_input = model.pos_enc(mel_input)
            dec = mel_input
            for layer in model.decoder_layers:
                dec = layer(dec)
            mel_pred = model.mel_linear(dec)
        mel_raw = mel_pred[0].numpy() * mel_std + mel_mean
        audio = voc.run(None, {'logmel': mel_raw.astype(np.float32)})[0].flatten()
        wav = str(OUTDIR / (pid + '.wav'))
        sf.write(wav, audio, 24000)
        asr_text, _ = transcribe(asr, wav)
        cer, _, _, _, _ = cer_detail(text, asr_text)
        results.append(cer)
        if idx < 5 or idx % 10 == 0:
            print("  [%d/30] %s: CER=%.0f%% | %s" % (idx+1, pid, cer*100, text[:25]))
    cers = np.array(results)
    print("\n=== v8 full_e2e pred-all CER ===")
    print("Mean CER: %.1f%%" % (cers.mean()*100))
    print("100%% CER: %d/30" % sum(1 for c in cers if c >= 0.99))
    print("<15%% CER: %d/30" % sum(1 for c in cers if c < 0.15))
    return cers.mean()*100

pred_cer = eval_pred_all()
print("\n" + "="*60)
print("=== v8 Full E2E Summary ===")
print("  Teacher ceiling:  20.2%%")
print("  v7 GT dur:        23.9%%")
print("  v7 pred-all:      96.9%%")
print("  v8 pred-all:      %.1f%%" % pred_cer)
print("="*60)
