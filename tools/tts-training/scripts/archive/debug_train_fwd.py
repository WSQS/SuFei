"""Quick test: replicate EXACT training forward for eval to check if train/eval mismatch is the cause."""
import json, sys, math, random, numpy as np, torch, soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))
from nar_fastspeech2 import FastSpeech2
from nar_train import length_regulate_batch  # import training LR
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
model.load_state_dict(ckpt['model'])
model.eval()

train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v2.jsonl', encoding='utf-8')]
random.seed(42)
sample = random.sample(train, 5)

asr = load_asr_model()

def train_forward_eval(model, phone_ids, durations_gt, mel_gt_len, f0_gt, energy_gt, stats):
    """Exact replication of training default E2E forward pass."""
    mel_mean_t = torch.tensor(stats['mel_mean'], dtype=torch.float32)
    mel_std_t = torch.tensor(stats['mel_std'], dtype=torch.float32)
    f0_norm = torch.where(f0_gt > 0,
        (torch.log(f0_gt.clamp(min=1)) - stats['f0_mean']) / stats['f0_std'],
        torch.zeros_like(f0_gt))
    energy_norm = (energy_gt - stats['energy_mean']) / stats['energy_std']

    # Forward
    x = model.embedding(phone_ids) * math.sqrt(model.d_model)
    x = model.pos_enc(x)
    for layer in model.encoder_layers:
        x = layer(x)

    pitch_pred_enc = model.pitch_predictor(x)
    energy_pred_enc = model.energy_predictor(x)

    # Default E2E: GT durations + predicted pitch/energy
    mel_input = length_regulate_batch(x, durations_gt)
    T_pred = mel_input.size(1)

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
    return mel_pred


OUTDIR = ROOT / 'output' / 'debug_train_fwd'
OUTDIR.mkdir(parents=True, exist_ok=True)
cers = []
for rec in sample:
    pid = rec['poem_id']
    text = rec['text']
    npz = np.load(FEAT_DIR / f"{pid}.npz")

    phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)
    dur_gt = torch.tensor([rec['durations']], dtype=torch.long)
    f0_gt = torch.tensor([npz['f0']], dtype=torch.float)
    energy_gt = torch.tensor([npz['energy']], dtype=torch.float)

    with torch.no_grad():
        mel_pred = train_forward_eval(model, phone_ids, dur_gt, rec['mel_len'], f0_gt, energy_gt, STATS)

    mel_raw = mel_pred[0].numpy() * mel_std + mel_mean
    audio = voc.run(None, {'logmel': mel_raw.astype(np.float32)})[0].flatten()
    wav = str(OUTDIR / f"{pid}.wav")
    sf.write(wav, audio, 24000)
    asr_text, _ = transcribe(asr, wav)
    cer, _, _, _, _ = cer_detail(text, asr_text)
    cers.append(cer)

    # Also check L1
    mel_gt_norm = (npz['mel'] - mel_mean) / mel_std
    T = min(mel_pred.shape[1], mel_gt_norm.shape[0])
    l1 = np.abs(mel_pred[0, :T].numpy() - mel_gt_norm[:T]).mean()

    print("  %s: L1=%.4f, CER=%.0f%% | ASR: %s" % (pid, l1, cer*100, asr_text[:30]))

print("\nMean train-forward CER: %.1f%%" % (np.mean(cers)*100))
print("(If this is also ~100%, the problem is NOT train/eval mismatch)")
