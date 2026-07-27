"""Check duration predictor output at step 12000."""
import json, torch, numpy as np, sys
sys.path.insert(0, 'tools/tts-training/scripts')
from nar_fastspeech2 import FastSpeech2

ckpt = torch.load('tools/tts-training/checkpoints/fs2_step12000.pt', map_location='cpu', weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt['model'])
model.eval()

manifest = [json.loads(l) for l in open('tools/tts-training/data/train_manifest.jsonl', encoding='utf-8')]
rec = manifest[0]

ids = torch.tensor([rec['phoneme_ids']])
with torch.no_grad():
    _, log_dur, _, _ = model(ids)

ld = log_dur[0]
print(f"log_dur: min={ld.min():.3f}, max={ld.max():.3f}, mean={ld.mean():.3f}, median={ld.median():.3f}")
print(f"exp(mean)={torch.exp(torch.tensor(ld.mean())):.1f}, exp(median)={torch.exp(torch.tensor(ld.median())):.1f}")
print(f"Bias: {model.duration_predictor.linear.bias.item():.4f}")
print(f"First 20 log_dur: {ld[:20].tolist()}")
print(f"First 20 dur: {[f'{torch.exp(x):.1f}' for x in ld[:20]]}")

# Check GT
gt = np.array(rec['durations'])
print(f"\nGT durations: min={gt.min()}, max={gt.max()}, mean={gt.mean():.1f}, sum={gt.sum()}")
print(f"Pred durations sum: {torch.exp(ld).sum():.0f}")
print(f"Ratio: {torch.exp(ld).sum().item()/gt.sum():.2%}")

# Quick fix: just scale the bias
print(f"\nIf we multiply bias by 2.0:")
new_bias = model.duration_predictor.linear.bias.item() * 2.0
print(f"  New bias={new_bias:.3f}, exp={torch.exp(torch.tensor(new_bias)):.1f}")
