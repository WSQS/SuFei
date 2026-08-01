"""M1: Evaluate MAS alignment quality on the frozen v9 encoder.

Tests whether the v9 encoder produces representations that MAS can align
well. If MAS durations correlate highly with teacher durations on the
frozen encoder, we can confidently proceed to M2 (full MAS training).

Two evaluation passes:
  Pass 1: Random projections (baseline — what MAS gets with no training)
  Pass 2: Trained projections (after a few steps of alignment optimization)
"""
import json, sys, math, time, numpy as np, torch
import torch.nn.functional as F
from pathlib import Path

# Force unbuffered output for SSH
import functools
print = functools.partial(print, flush=True)

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))

from nar_fastspeech2 import FastSpeech2
from mas import AlignmentModule, maximum_path_np, mas_durations

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ─── Load v9 model ─────────────────────────────────────────────────────

ckpt = torch.load(str(ROOT / 'checkpoints' / 'fs2_final.pt'), map_location=device, weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0).to(device)
state_dict = {k: v for k, v in ckpt['model'].items() if not k.startswith('film_gen')}
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded v9 (step={ckpt.get('step')})")

# ─── Load data ─────────────────────────────────────────────────────────

FEAT_DIR = ROOT / 'data' / 'paddle_distill_features_v3'
STATS = json.load(open(ROOT / 'data' / 'train_300_norm_stats_v3.json'))
mel_mean = torch.tensor(STATS['mel_mean'], dtype=torch.float32, device=device)
mel_std = torch.tensor(STATS['mel_std'], dtype=torch.float32, device=device)

manifests = {
    'train': ROOT / 'data' / 'train_300_manifest_v3.jsonl',
    'holdout': ROOT / 'data' / 'holdout_20_manifest_v3.jsonl',
}

records = {}
for name, path in manifests.items():
    records[name] = [json.loads(l) for l in open(path, encoding='utf-8')]
    print(f"  {name}: {len(records[name])} samples")


# ─── Helper: get encoder output ────────────────────────────────────────

@torch.no_grad()
def get_encoder_out(phone_ids):
    """Get encoder output for a single sample.
    Returns: [1, L, D_model]
    """
    x = model.embedding(phone_ids) * math.sqrt(model.d_model)
    x = model.pos_enc(x)
    for layer in model.encoder_layers:
        x = layer(x)
    return x


# ─── Pass 1: Random projections ────────────────────────────────────────

print("\n" + "="*60)
print("Pass 1: Random projections (untrained alignment)")
print("="*60)

align_module = AlignmentModule(d_model=256, n_mels=80, d_align=128).to(device)

def evaluate_alignment(align_mod, sample_records, label):
    """Evaluate MAS alignment quality against teacher durations."""
    teacher_durs_all = []
    mas_durs_all = []
    ratios = []
    correlations = []
    per_sample = []

    for idx, rec in enumerate(sample_records):
        pid = rec['poem_id']
        phone_ids = torch.tensor([rec['phoneme_ids']], dtype=torch.long, device=device)
        teacher_dur = np.array(rec['durations'], dtype=np.float32)
        L = len(rec['phoneme_ids'])

        if idx % 10 == 0:
            print(f"  [{label}] {idx}/{len(sample_records)}...")

        npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
        mel = torch.tensor(npz['mel'].astype(np.float32)).unsqueeze(0).to(device)
        mel_norm = (mel - mel_mean) / mel_std
        T = mel_norm.size(1)

        # Skip if too short
        if T < L:
            continue

        # Get encoder output
        with torch.no_grad():
            enc_out = get_encoder_out(phone_ids)
            log_prob = align_mod.compute_log_prob(enc_out, mel_norm)  # [1, L, T]

        # Run MAS
        cost = log_prob[0, :L, :T].detach().cpu().numpy().astype(np.float64)
        mas_dur = mas_durations(cost).astype(np.float32)

        # Metrics
        teacher_dur_clipped = teacher_dur
        mas_dur_clipped = mas_dur

        # Per-phoneme correlation
        if len(mas_dur) > 1:
            corr = np.corrcoef(teacher_dur_clipped, mas_dur_clipped)[0, 1]
            correlations.append(corr)

        # Ratio
        dur_ratio = mas_dur.sum() / max(teacher_dur.sum(), 1)
        ratios.append(dur_ratio)

        per_sample.append({
            'pid': pid,
            'L': L,
            'T': T,
            'corr': corr if len(mas_dur) > 1 else 0,
            'ratio': dur_ratio,
            'teacher_dur': teacher_dur.astype(int).tolist(),
            'mas_dur': mas_dur.astype(int).tolist(),
        })

        teacher_durs_all.extend(teacher_dur_clipped.tolist())
        mas_durs_all.extend(mas_dur_clipped.tolist())

    teacher_durs_all = np.array(teacher_durs_all)
    mas_durs_all = np.array(mas_durs_all)

    # Aggregate stats
    mean_corr = np.mean(correlations) if correlations else 0
    mean_ratio = np.mean(ratios) if ratios else 0
    global_corr = np.corrcoef(teacher_durs_all, mas_durs_all)[0, 1] if len(teacher_durs_all) > 2 else 0

    # Duration distribution
    mas_arr = mas_durs_all
    teacher_arr = teacher_durs_all

    print(f"\n  [{label}] {len(per_sample)} samples")
    print(f"  Per-sample Pearson corr: mean={mean_corr:.3f}, median={np.median(correlations):.3f}")
    print(f"  Global Pearson corr:     {global_corr:.3f}")
    print(f"  MAS dur ratio (MAS/GT):  mean={mean_ratio:.3f}")
    print(f"  MAS dur distribution:    mean={mas_arr.mean():.1f}, median={np.median(mas_arr):.0f}, "
          f"min={mas_arr.min():.0f}, max={mas_arr.max():.0f}")
    print(f"  Teacher dur distribution: mean={teacher_arr.mean():.1f}, median={np.median(teacher_arr):.0f}, "
          f"min={teacher_arr.min():.0f}, max={teacher_arr.max():.0f}")
    print(f"  MAS dur<=2: {(mas_arr <= 2).sum()/len(mas_arr)*100:.1f}%  "
          f"(teacher: {(teacher_arr <= 2).sum()/len(teacher_arr)*100:.1f}%)")

    # Per-sample L1 on log duration
    log_l1s = []
    for s in per_sample:
        td = np.array(s['teacher_dur'], dtype=np.float32)
        md = np.array(s['mas_dur'], dtype=np.float32)
        log_l1 = np.abs(np.log(td + 1) - np.log(md + 1)).mean()
        log_l1s.append(log_l1)
    print(f"  Per-sample log-dur L1:   mean={np.mean(log_l1s):.3f}")

    # Show a few examples
    print(f"\n  Sample examples:")
    for s in per_sample[:5]:
        print(f"    {s['pid']}: corr={s['corr']:.3f}, ratio={s['ratio']:.3f}, "
              f"mas={s['mas_dur'][:10]}{'...' if len(s['mas_dur'])>10 else ''}  "
              f"gt={s['teacher_dur'][:10]}{'...' if len(s['teacher_dur'])>10 else ''}")

    return {
        'mean_corr': mean_corr,
        'global_corr': global_corr,
        'mean_ratio': mean_ratio,
        'per_sample': per_sample,
    }


# Evaluate random projections
random_train = evaluate_alignment(align_module, records['train'][:30], "random/train")
random_holdout = evaluate_alignment(align_module, records['holdout'][:10], "random/holdout")


# ─── Pass 2: Quick projection training ─────────────────────────────────

print("\n" + "="*60)
print("Pass 2: Training projections (alignment-only optimization)")
print("="*60)

# Prepare training data from train set
train_data = []
for rec in records['train']:
    pid = rec['poem_id']
    phone_ids_list = rec['phoneme_ids']
    npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
    mel_arr = npz['mel'].astype(np.float32)
    L = len(phone_ids_list)
    T = mel_arr.shape[0]
    if T < L:
        continue
    train_data.append({
        'phone_ids': torch.tensor([phone_ids_list], dtype=torch.long, device=device),
        'mel': torch.tensor(mel_arr).unsqueeze(0).to(device),
        'L': L,
        'T': T,
    })

print(f"Training data: {len(train_data)} samples")

# Fresh alignment module
align_module_trained = AlignmentModule(d_model=256, n_mels=80, d_align=128).to(device)
optimizer = torch.optim.Adam(align_module_trained.parameters(), lr=1e-3)

# Train only the projection layers (encoder is frozen)
N_STEPS = 2000
BATCH_SIZE = 8
t_start = time.time()

for step in range(1, N_STEPS + 1):
    # Sample a mini-batch
    indices = np.random.choice(len(train_data), BATCH_SIZE, replace=False)
    batch_items = [train_data[i] for i in indices]

    total_loss = 0
    for item in batch_items:
        phone_ids = item['phone_ids']
        mel = item['mel']
        L = item['L']
        T = item['T']

        mel_norm = (mel - mel_mean) / mel_std

        with torch.no_grad():
            enc_out = get_encoder_out(phone_ids)

        log_prob = align_module_trained.compute_log_prob(enc_out, mel_norm)  # [1, L, T]

        # Get MAS path (detached — no grad through DP)
        cost = log_prob[0, :L, :T].detach().cpu().numpy().astype(np.float64)
        path = maximum_path_np(cost)  # [L, T]
        path_t = torch.tensor(path, dtype=torch.float32, device=device)

        # Alignment loss: -mean(log_prob * path)
        phone_mask = torch.zeros(1, L, dtype=torch.bool, device=device)
        mel_mask = torch.zeros(1, T, dtype=torch.bool, device=device)
        loss = align_module_trained.alignment_loss(
            log_prob[:, :L, :T], path_t.unsqueeze(0), phone_mask, mel_mask
        )
        total_loss += loss

    total_loss = total_loss / BATCH_SIZE
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 200 == 0 or step == 1:
        elapsed = time.time() - t_start
        print(f"  step {step}/{N_STEPS} | loss={total_loss.item():.4f} | {step/elapsed:.1f} step/s")


# Evaluate trained projections
trained_train = evaluate_alignment(align_module_trained, records['train'][:30], "trained/train")
trained_holdout = evaluate_alignment(align_module_trained, records['holdout'][:10], "trained/holdout")


# ─── Summary ────────────────────────────────────────────��──────────────

print("\n" + "="*60)
print("M1 SUMMARY")
print("="*60)
print(f"\n{'Metric':<30} {'Random':>10} {'Trained':>10} {'Delta':>10}")
print("-" * 62)
print(f"{'Train per-sample corr':<30} {random_train['mean_corr']:>10.3f} {trained_train['mean_corr']:>10.3f} {trained_train['mean_corr']-random_train['mean_corr']:>+10.3f}")
print(f"{'Holdout per-sample corr':<30} {random_holdout['mean_corr']:>10.3f} {trained_holdout['mean_corr']:>10.3f} {trained_holdout['mean_corr']-random_holdout['mean_corr']:>+10.3f}")
print(f"{'Train global corr':<30} {random_train['global_corr']:>10.3f} {trained_train['global_corr']:>10.3f} {trained_train['global_corr']-random_train['global_corr']:>+10.3f}")
print(f"{'Holdout global corr':<30} {random_holdout['global_corr']:>10.3f} {trained_holdout['global_corr']:>10.3f} {trained_holdout['global_corr']-random_holdout['global_corr']:>+10.3f}")
print(f"{'Train dur ratio':<30} {random_train['mean_ratio']:>10.3f} {trained_train['mean_ratio']:>10.3f} {trained_train['mean_ratio']-random_train['mean_ratio']:>+10.3f}")

print(f"\nDecision threshold:")
best_corr = max(trained_train['mean_corr'], trained_holdout['mean_corr'])
if best_corr > 0.9:
    print(f"  PASS — best corr={best_corr:.3f} > 0.9. Proceed to M2.")
elif best_corr > 0.7:
    print(f"  MARGINAL — best corr={best_corr:.3f}. MAS can work but needs more training.")
else:
    print(f"  FAIL — best corr={best_corr:.3f} < 0.7. Encoder representations may be insufficient.")

# Save alignment module state for M2 warm-start
if best_corr > 0.7:
    save_path = ROOT / 'checkpoints' / 'mas_align_v9_m1.pt'
    torch.save({
        'align_module': align_module_trained.state_dict(),
        'train_corr': trained_train['mean_corr'],
        'holdout_corr': trained_holdout['mean_corr'],
    }, str(save_path))
    print(f"\nSaved alignment module: {save_path}")
