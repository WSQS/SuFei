import torch, json, numpy as np, sys, math
from pathlib import Path
sys.path.insert(0, 'tools/tts-training/scripts')
from nar_fastspeech2 import FastSpeech2

ROOT = Path('tools/tts-training')
ckpt = torch.load(str(ROOT / 'checkpoints' / 'e2e_v3_step12k.pt'), map_location='cpu', weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt['model'])
model.eval()

train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest.jsonl', encoding='utf-8')]
rec = train[0]
phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long)

with torch.no_grad():
    mel_out, log_dur, pp, ep = model.forward(phone_ids)

dur_pred = np.round(np.exp(log_dur[0].numpy())).clip(min=1)
print(f'v3 step12k poem_0001:')
print(f'  mel shape: {mel_out.shape}')
print(f'  pred_dur sum: {dur_pred.sum():.0f} vs gt: {sum(rec["durations"])}')
print(f'  pitch pred: mean={pp.mean():.3f} std={pp.std():.3f}')
print(f'  energy pred: mean={ep.mean():.3f} std={ep.std():.3f}')
print(f'  mel pred: mean={mel_out.mean():.3f} std={mel_out.std():.3f}')

ckpt2 = torch.load(str(ROOT / 'checkpoints' / 'FullE2E_v2_step24000_slim.pt'), map_location='cpu', weights_only=False)
model2 = FastSpeech2(vocab_size=268, dropout=0.0)
model2.load_state_dict(ckpt2['model'])
model2.eval()

with torch.no_grad():
    mel_out2, log_dur2, pp2, ep2 = model2.forward(phone_ids)

dur_pred2 = np.round(np.exp(log_dur2[0].numpy())).clip(min=1)
print(f'\nv2 step24k poem_0001:')
print(f'  mel shape: {mel_out2.shape}')
print(f'  pred_dur sum: {dur_pred2.sum():.0f}')
print(f'  pitch pred: mean={pp2.mean():.3f} std={pp2.std():.3f}')
print(f'  energy pred: mean={ep2.mean():.3f} std={ep2.std():.3f}')
print(f'  mel pred: mean={mel_out2.mean():.3f} std={mel_out2.std():.3f}')
