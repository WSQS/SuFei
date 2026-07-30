"""Phase 2 diagnostics: ceiling, duration predictor, mel roundtrip."""
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
FS2_DIR = ROOT / 'models' / 'paddlespeech_onnx' / 'fastspeech2_csmsc_onnx_0.2.0'
FEAT_DIR = ROOT / 'data' / 'paddle_distill_features_v2'
STATS = json.load(open(ROOT / 'data' / 'train_300_norm_stats_v2.json'))
mel_mean = np.array(STATS['mel_mean'], dtype=np.float32)
mel_std = np.array(STATS['mel_std'], dtype=np.float32)

fs2 = ort.InferenceSession(str(FS2_DIR / 'fastspeech2_csmsc.onnx'), providers=['CPUExecutionProvider'])
voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'), providers=['CPUExecutionProvider'])

ckpt = torch.load(str(ROOT / 'checkpoints' / 'fs2_final.pt'), map_location='cpu', weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
state_dict = {k: v for k, v in ckpt['model'].items() if not k.startswith('film_gen')}
model.load_state_dict(state_dict)
model.eval()

train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v2.jsonl', encoding='utf-8')]
random.seed(42)
sample = random.sample(train, 30)

OUTDIR = ROOT / 'output' / 'diag_v7'
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
print("="*60)
print("CHECK 1: Mel roundtrip — on-the-fly vs stored npz vs teacher")
print("="*60)
rec = train[0]
pid = rec['poem_id']
phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)
mel_live = fs2.run(None, {"text": phone_ids})[0]
if mel_live.ndim == 3: mel_live = mel_live[0]
npz = np.load(FEAT_DIR / f"{pid}.npz")
mel_stored = npz['mel']
print("  live vs stored identical: %s (L1=%.6f)" % (
    np.allclose(mel_live, mel_stored, atol=1e-5),
    np.abs(mel_live - mel_stored).mean()))

# Audio from live vs stored
audio_live = voc.run(None, {'logmel': mel_live.astype(np.float32)})[0].flatten()
audio_stored = voc.run(None, {'logmel': mel_stored.astype(np.float32)})[0].flatten()
print("  audio live vs stored max abs diff: %.8f" % np.abs(audio_live - audio_stored).max())

# ============================================================
print("\n" + "="*60)
print("CHECK 2: Teacher CER on same 30 samples (ceiling check)")
print("="*60)
asr = load_asr_model()
teacher_cers = []
for idx, rec in enumerate(sample):
    pid = rec['poem_id']
    text = rec['text']
    phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)
    teacher_mel = fs2.run(None, {"text": phone_ids})[0]
    if teacher_mel.ndim == 3: teacher_mel = teacher_mel[0]
    audio = voc.run(None, {'logmel': teacher_mel.astype(np.float32)})[0].flatten()
    wav = str(OUTDIR / f"teacher_{pid}.wav")
    sf.write(wav, audio, 24000)
    asr_text, _ = transcribe(asr, wav)
    cer, _, _, _, _ = cer_detail(text, asr_text)
    teacher_cers.append(cer)
    if idx < 5:
        print("  [%d] %s: CER=%.0f%% ref=%s hyp=%s" % (idx, pid, cer*100, text[:20], asr_text[:20]))

print("\n  Teacher mean CER: %.1f%% (n=30)" % (np.mean(teacher_cers)*100))

# ============================================================
print("\n" + "="*60)
print("CHECK 3: Duration predictor analysis")
print("="*60)
dur_ratios = []
dur_l1_phoneme = []
log_dur_pred_all = []
log_dur_gt_all = []
zero_dur_count = 0
total_phonemes = 0

for rec in sample:
    pid = rec['poem_id']
    phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)
    dur_gt = rec['durations']

    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        log_dur = model.duration_predictor(x)

    pred_dur = log_dur.exp().round().clamp(min=1).long()[0].numpy()
    gt_dur = np.array(dur_gt)

    ratio = pred_dur.sum() / max(gt_dur.sum(), 1)
    dur_ratios.append(ratio)

    # Per-phoneme L1
    log_pred = log_dur[0].numpy()
    log_gt = np.log(np.clip(gt_dur, 1, 100).astype(float))
    dur_l1_phoneme.append(np.abs(log_pred - log_gt).mean())

    # Count zero-duration predictions (before clamp)
    raw_pred = log_dur.exp().round()[0].numpy()
    zero_dur_count += (raw_pred <= 0).sum()
    total_phonemes += len(raw_pred)

    log_dur_pred_all.extend(log_pred)
    log_dur_gt_all.extend(log_gt)

dur_ratios = np.array(dur_ratios)
print("  Duration ratio (pred_sum/gt_sum):")
print("    mean=%.2f, std=%.2f, range=[%.2f, %.2f]" % (
    dur_ratios.mean(), dur_ratios.std(), dur_ratios.min(), dur_ratios.max()))
print("  Per-phoneme log-dur L1: mean=%.3f" % np.mean(dur_l1_phoneme))
print("  Zero-duration predictions (pre-clamp): %d/%d (%.1f%%)" % (
    zero_dur_count, total_phonemes, zero_dur_count/total_phonemes*100))

# Correlation
log_dur_pred_all = np.array(log_dur_pred_all)
log_dur_gt_all = np.array(log_dur_gt_all)
corr = np.corrcoef(log_dur_pred_all, log_dur_gt_all)[0, 1]
print("  Log-dur correlation: %.3f" % corr)

# Distribution comparison
print("\n  GT duration distribution:")
gt_durs = np.array([d for r in sample for d in r['durations']])
print("    median=%d, mean=%.1f, dur<=2: %.1f%%, dur<=5: %.1f%%" % (
    np.median(gt_durs), gt_durs.mean(),
    (gt_durs <= 2).sum()/len(gt_durs)*100,
    (gt_durs <= 5).sum()/len(gt_durs)*100))
print("  Pred duration distribution:")
pred_durs_all = []
for rec in sample:
    phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)
    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        log_dur = model.duration_predictor(x)
    pred_durs_all.extend(log_dur.exp().round().clamp(min=1).long()[0].numpy())
pred_durs_all = np.array(pred_durs_all)
print("    median=%d, mean=%.1f, dur<=2: %.1f%%, dur<=5: %.1f%%" % (
    np.median(pred_durs_all), pred_durs_all.mean(),
    (pred_durs_all <= 2).sum()/len(pred_durs_all)*100,
    (pred_durs_all <= 5).sum()/len(pred_durs_all)*100))

# ============================================================
print("\n" + "="*60)
print("CHECK 4: Duration label source")
print("="*60)
print("  v2 manifest durations are re-adjusted from old MFA TextGrid durations.")
print("  They were NOT dumped from teacher FS2 internal states.")
print("  rebuild_data_v2.py: adjusted old_durations to match teacher_mel length.")
print("  This means duration labels may still have MFA alignment errors.")

# Show 3 examples
for rec in sample[:3]:
    pid = rec['poem_id']
    dur = rec['durations']
    n_phones = len(dur)
    mel_len = rec['mel_len']
    print("  %s: %d phones, mel_len=%d, sum(dur)=%d" % (pid, n_phones, mel_len, sum(dur)))
    print("    dur[:10]=%s" % str(dur[:10]))
