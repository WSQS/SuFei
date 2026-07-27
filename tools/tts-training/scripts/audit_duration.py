"""Audit duration predictor after training."""
import json, numpy as np, torch, sys
sys.path.insert(0, 'tools/tts-training/scripts')
from nar_fastspeech2 import FastSpeech2

ckpt = torch.load('tools/tts-training/checkpoints/fs2_step6000.pt', map_location='cpu', weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt['model'])
model.eval()

w = model.duration_predictor.linear.weight
b = model.duration_predictor.linear.bias
print(f"Linear: weight mean={w.mean():.6f} std={w.std():.6f}, bias={b.item():.4f} (exp={torch.exp(b).item():.1f})")

manifest = [json.loads(l) for l in open('tools/tts-training/data/train_manifest.jsonl', encoding='utf-8')]

for i, rec in enumerate(manifest[:3]):
    ids = torch.tensor([rec['phoneme_ids']])
    gt_dur = np.array(rec['durations'])
    with torch.no_grad():
        x = model.embedding(ids) * (256**0.5)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        log_dur = model.duration_predictor(x)
    pred = log_dur[0].exp()

    print(f"\n{rec['poem_id']}")
    print(f"  GT: sum={gt_dur.sum()}, mean={gt_dur.mean():.1f}, len={len(gt_dur)}")
    print(f"  Pred: sum={pred.sum():.0f}, mean={pred.mean():.1f}, ratio={pred.sum().item()/gt_dur.sum():.2%}")
    print(f"  log_dur: mean={log_dur[0].mean():.3f} std={log_dur[0].std():.3f} min={log_dur[0].min():.3f} max={log_dur[0].max():.3f}")
    gt_log = np.log(np.clip(gt_dur, 1, None))
    print(f"  GT log:  mean={gt_log.mean():.3f} std={gt_log.std():.3f} min={gt_log.min():.3f} max={gt_log.max():.3f}")

# Check training loss formula
print("\n=== Training formula check ===")
print("Training: log_dur_gt = torch.log(durations.float().clamp(min=1, max=100))")
print("Inference: durations = log_pred_durations.exp().round().clamp(min=0).long()")
print("exp(log(dur)) == dur: symmetric")
print(f"clamp(1,100) on GT: max 74 in sample, max dur in data: {max(np.array(rec['durations']))}")

# Check what the predictor actually outputs
print(f"\nPredictor raw output mean: {log_dur.mean():.3f}")
print(f"Bias: {b.item():.3f}")
print(f"Conv contribution mean: {log_dur.mean().item() - b.item():.3f}")

# Per-phoneme breakdown
print("\n=== Per-phoneme breakdown (first 15) ===")
for j in range(min(15, len(gt_dur))):
    print(f"  ph{j:2d}: GT={gt_dur[j]:3d} pred={pred[j]:5.1f} log_pred={log_dur[0,j]:.3f} log_gt={np.log(max(gt_dur[j],1)):.3f}")