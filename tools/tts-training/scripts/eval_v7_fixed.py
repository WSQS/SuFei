"""Phase 2 eval v2: use TRAINING forward path (GT dur + pred pitch/energy = training condition)."""
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
f0_mean = STATS['f0_mean']
f0_std = STATS['f0_std']
energy_mean = STATS['energy_mean']
energy_std = STATS['energy_std']

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

asr = load_asr_model()


def train_forward(phone_ids, durations_gt, use_gt_variance=False, f0_gt=None, energy_gt=None):
    """Exact replication of training default E2E forward."""
    x = model.embedding(phone_ids) * math.sqrt(model.d_model)
    x = model.pos_enc(x)
    for layer in model.encoder_layers:
        x = layer(x)

    log_dur_pred = model.duration_predictor(x)
    pitch_pred_enc = model.pitch_predictor(x)
    energy_pred_enc = model.energy_predictor(x)

    mel_input = length_regulate_batch(x, durations_gt)
    T_pred = mel_input.size(1)

    if use_gt_variance and f0_gt is not None:
        T_var = min(f0_gt.size(1), T_pred)
        pitch_expanded = f0_gt[:, :T_var]
        energy_expanded = energy_gt[:, :T_var]
    else:
        pitch_expanded = length_regulate_batch(
            pitch_pred_enc.unsqueeze(-1), durations_gt).squeeze(-1)
        energy_expanded = length_regulate_batch(
            energy_pred_enc.unsqueeze(-1), durations_gt).squeeze(-1)

    pitch_embed = model.pitch_embed(pitch_expanded[:, :T_pred].unsqueeze(-1))
    energy_embed = model.energy_embed(energy_expanded[:, :T_pred].unsqueeze(-1))

    mel_input = mel_input + pitch_embed + energy_embed
    mel_input = model.pos_enc(mel_input)

    dec = mel_input
    for layer in model.decoder_layers:
        dec = layer(dec)
    mel_pred = model.mel_linear(dec)
    return mel_pred, log_dur_pred


def eval_condition(name, use_gt_variance=False):
    OUTDIR = ROOT / 'output' / ('v7_clean_24k_' + name)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, rec in enumerate(sample):
        pid = rec['poem_id']
        phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)
        text = rec['text']
        npz = np.load(FEAT_DIR / f"{pid}.npz")
        dur_gt = torch.tensor([rec['durations']], dtype=torch.long)

        f0_gt = energy_gt = None
        if use_gt_variance:
            f0_raw = torch.tensor([npz['f0']], dtype=torch.float)
            e_raw = torch.tensor([npz['energy']], dtype=torch.float)
            f0_gt = torch.where(f0_raw > 0,
                (torch.log(f0_raw.clamp(min=1)) - f0_mean) / f0_std,
                torch.zeros_like(f0_raw))
            energy_gt = (e_raw - energy_mean) / energy_std

        with torch.no_grad():
            mel_pred, _ = train_forward(phone_ids, dur_gt, use_gt_variance, f0_gt, energy_gt)

        mel_raw = mel_pred[0].numpy() * mel_std + mel_mean
        audio = voc.run(None, {'logmel': mel_raw.astype(np.float32)})[0].flatten()
        wav = str(OUTDIR / (pid + '.wav'))
        sf.write(wav, audio, 24000)
        asr_text, _ = transcribe(asr, wav)
        cer, _, _, _, _ = cer_detail(text, asr_text)
        results.append(cer)
        if idx < 3 or idx % 10 == 0:
            print("  [%d/30] %s: CER=%.0f%% | %s" % (idx+1, pid, cer*100, text[:25]))

    cers = np.array(results)
    print("\n=== v7 %s CER ===" % name)
    print("Mean CER: %.1f%%" % (cers.mean()*100))
    print("100%% CER: %d/30" % sum(1 for c in cers if c >= 0.99))
    print("<15%% CER: %d/30" % sum(1 for c in cers if c < 0.15))
    return cers.mean()*100


# 3 conditions
print("\n--- A: GT dur + pred pitch/energy (training condition) ---")
train_cer = eval_condition('train_cond')

print("\n--- B: GT dur + GT pitch/energy (GT variance) ---")
gt_cer = eval_condition('gt_var', use_gt_variance=True)

print("\n--- C: pred dur + pred pitch/energy (full inference) ---")
# For full inference, need to use predicted durations
def eval_pred_all():
    OUTDIR = ROOT / 'output' / 'v7_clean_24k_pred_all'
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
        if idx < 3 or idx % 10 == 0:
            print("  [%d/30] %s: CER=%.0f%% | %s" % (idx+1, pid, cer*100, text[:25]))

    cers = np.array(results)
    print("\n=== v7 pred-all CER ===")
    print("Mean CER: %.1f%%" % (cers.mean()*100))
    print("100%% CER: %d/30" % sum(1 for c in cers if c >= 0.99))
    print("<15%% CER: %d/30" % sum(1 for c in cers if c < 0.15))
    return cers.mean()*100

pred_cer = eval_pred_all()

print("\n" + "="*60)
print("=== v7 Clean Classic FS2 Summary ===")
print("  Teacher mel ceiling: ~20.2%%")
print("  A: GT dur + pred var (train cond):  %.1f%%" % train_cer)
print("  B: GT dur + GT var:                 %.1f%%" % gt_cer)
print("  C: pred-all (full inference):       %.1f%%" % pred_cer)
print("="*60)
