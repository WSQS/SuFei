"""Evaluate v10 (MAS-trained FS2) on train 30 + holdout 20.

Identical evaluation protocol as eval_v9_holdout.py:
  pred-all: predicted duration + predicted pitch/energy (full inference)
  Uses the same ASR + CER pipeline.

The v10 model has the same inference architecture as v9 — the AlignmentModule
is training-only and discarded. We just load model weights (no align_module).
"""
import json, sys, math, random, numpy as np, torch, soundfile as sf
import onnxruntime as ort
from pathlib import Path

# Optional inference-time duration scale compensation, e.g.:
#   python eval_v10.py --dur_scale 1.25
DUR_SCALE = 1.0
if '--dur_scale' in sys.argv:
    DUR_SCALE = float(sys.argv[sys.argv.index('--dur_scale') + 1])
print(f"DUR_SCALE = {DUR_SCALE}")

CKPT_OVERRIDE = None
if '--ckpt' in sys.argv:
    CKPT_OVERRIDE = sys.argv[sys.argv.index('--ckpt') + 1]

STATS_OVERRIDE = None
if '--stats' in sys.argv:
    STATS_OVERRIDE = sys.argv[sys.argv.index('--stats') + 1]

MANIFEST_SUFFIX = 'v3'  # default: v3 manifests
if '--m3' in sys.argv:
    MANIFEST_SUFFIX = 'm3'

HOLDOUT_MANIFEST_OVERRIDE = None
if '--holdout_manifest' in sys.argv:
    HOLDOUT_MANIFEST_OVERRIDE = sys.argv[sys.argv.index('--holdout_manifest') + 1]
TRAIN_MANIFEST_OVERRIDE = None
if '--train_manifest' in sys.argv:
    TRAIN_MANIFEST_OVERRIDE = sys.argv[sys.argv.index('--train_manifest') + 1]

OUTDIR_NAME = 'v10_holdout'
if '--outdir' in sys.argv:
    OUTDIR_NAME = sys.argv[sys.argv.index('--outdir') + 1]

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))
from nar_fastspeech2 import FastSpeech2
from nar_train import length_regulate_batch
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0'
FEAT_DIR = ROOT / 'data' / 'paddle_distill_features_v3'
STATS = json.load(open(STATS_OVERRIDE if STATS_OVERRIDE
                       else ROOT / 'data' / 'train_300_norm_stats_v3.json'))
mel_mean = np.array(STATS['mel_mean'], dtype=np.float32)
mel_std = np.array(STATS['mel_std'], dtype=np.float32)

voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'), providers=['CPUExecutionProvider'])

# Load v10 (or fallback to checkpoint path)
ckpt_path = Path(CKPT_OVERRIDE) if CKPT_OVERRIDE else ROOT / 'checkpoints' / 'v10_final.pt'
if not ckpt_path.exists():
    # Try latest step checkpoint
    import glob
    ckpts = sorted(glob.glob(str(ROOT / 'checkpoints' / 'v10_step*.pt')),
                   key=lambda f: int(f.split('step')[-1].split('.')[0]))
    if ckpts:
        ckpt_path = Path(ckpts[-1])
    else:
        print("No v10 checkpoint found!")
        sys.exit(1)

ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
sd = {k: v for k, v in ckpt['model'].items() if not k.startswith('film_gen')}
model.load_state_dict(sd)
model.eval()
print(f"Loaded {ckpt_path.name} (step={ckpt.get('step')})")

if MANIFEST_SUFFIX == 'm3':
    holdout_path = ROOT / 'data/m3_holdout_manifest.jsonl'
    train_path = ROOT / 'data/m3_train_manifest.jsonl'
else:
    holdout_path = ROOT / 'data/holdout_20_manifest_v3.jsonl'
    train_path = ROOT / 'data/train_300_manifest_v3.jsonl'
if HOLDOUT_MANIFEST_OVERRIDE:
    holdout_path = ROOT / HOLDOUT_MANIFEST_OVERRIDE
if TRAIN_MANIFEST_OVERRIDE:
    train_path = ROOT / TRAIN_MANIFEST_OVERRIDE
print(f"Manifests: holdout={holdout_path.name} train={train_path.name}")
holdout = [json.loads(l) for l in open(holdout_path, encoding='utf-8')]
train = [json.loads(l) for l in open(train_path, encoding='utf-8')]

asr = load_asr_model()

def infer_one(phone_ids):
    x = model.embedding(phone_ids) * math.sqrt(model.d_model)
    x = model.pos_enc(x)
    for layer in model.encoder_layers:
        x = layer(x)
    log_dur = model.duration_predictor(x)
    pitch_p = model.pitch_predictor(x)
    energy_p = model.energy_predictor(x)
    pred_dur = (log_dur.exp() * DUR_SCALE).round().clamp(min=1).long()
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
OUTDIR = ROOT / 'output' / OUTDIR_NAME
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

# --- Train (30 samples, same as v9 eval) ---
print("\n=== TRAIN (30 samples, same as v9 eval) ===")
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
print("  v10 (MAS) vs v9 (teacher dur)")
print("  Train CER:   %.1f%%  (v9: 24.0%%)" % (tc.mean()*100))
print("  Holdout CER: %.1f%%  (v9: 21.5%%)" % (hc.mean()*100))
print("  Gap: %.1fpp" % (hc.mean()*100 - tc.mean()*100))
print("="*60)

# Duration ratio
print("\nDuration analysis (train 30):")
ratios = []
for rec in sample:
    phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)
    teacher_dur = sum(rec['durations'])
    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        log_dur = model.duration_predictor(x)
        pred_dur = log_dur.exp().round().clamp(min=1).long()
    ratios.append(pred_dur.sum().item() / max(teacher_dur, 1))
print("  Duration ratio (pred/teacher): mean=%.3f (v9 was 0.79)" % np.mean(ratios))
