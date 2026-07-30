"""Debug: generate audio for 5 samples and see what ASR transcribes."""
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
model.load_state_dict(ckpt['model'])
model.eval()

train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v2.jsonl', encoding='utf-8')]
random.seed(42)
sample = random.sample(train, 5)

asr = load_asr_model()

for rec in sample:
    pid = rec['poem_id']
    phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)
    text = rec['text']
    npz = np.load(FEAT_DIR / f"{pid}.npz")

    # GT-all
    dur_gt = torch.tensor([rec['durations']], dtype=torch.long)
    pitch_gt = torch.tensor(npz['f0'], dtype=torch.float).unsqueeze(0)
    energy_gt = torch.tensor(npz['energy'], dtype=torch.float).unsqueeze(0)

    with torch.no_grad():
        mel_gt, _, _, _ = model.forward(phone_ids, durations=dur_gt, pitches=pitch_gt, energies=energy_gt)
        mel_pred, log_dur, _, _ = model.forward(phone_ids)

    # Denorm
    mel_gt_raw = mel_gt[0].numpy() * mel_std + mel_mean
    mel_pred_raw = mel_pred[0].numpy() * mel_std + mel_mean

    # Duration analysis
    pred_dur = log_dur.exp().round().clamp(min=0).long()[0]
    gt_dur = rec['durations']
    print("\n%s: %s" % (pid, text[:30]))
    print("  dur: gt_sum=%d, pred_sum=%d, ratio=%.2f" % (
        sum(gt_dur), pred_dur.sum().item(), pred_dur.sum().item() / max(sum(gt_dur), 1)))

    # Audio + ASR for both
    for name, mel_raw in [("GT-all", mel_gt_raw), ("pred-all", mel_pred_raw)]:
        audio = voc.run(None, {'logmel': mel_raw.astype(np.float32)})[0].flatten()
        wav_path = str(ROOT / 'output' / 'debug_v7' / f"{pid}_{name}.wav")
        Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, audio, 24000)
        asr_text, _ = transcribe(asr, wav_path)
        cer, _, _, _, _ = cer_detail(text, asr_text)
        print("  %s: CER=%.0f%% | ASR: %s" % (name, cer*100, asr_text[:40]))
