"""Diagnostic: trace layer-by-layer outputs for train vs holdout samples.

Picks one training sample and one holdout sample, runs both through the
model, and compares intermediate activations at each stage. This helps
isolate whether the failure is in the encoder, duration predictor,
length regulator, or decoder.

Usage:
  python diagnose_layers.py
  python diagnose_layers.py --ckpt checkpoints/D300e2e_step24000_slim.pt
"""
import argparse, json, math, sys, numpy as np, torch
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2

STATS_PATH = ROOT / "data" / "train_300_norm_stats.json"
FEAT_DIR = ROOT / "data" / "paddle_distill_features"
TRAIN_MANIFEST = ROOT / "data" / "train_300_manifest.jsonl"
HOLDOUT_MANIFEST = ROOT / "data" / "holdout_20_manifest.jsonl"

with open(STATS_PATH) as f:
    stats = json.load(f)


def fmt_tensor(t, name=""):
    if t.dim() == 0:
        return f"{name}: scalar={t.item():.4f}"
    flat = t.flatten().float()
    return (
        f"{name}: shape={list(t.shape)} "
        f"mean={flat.mean():.4f} std={flat.std():.4f} "
        f"min={flat.min():.4f} max={flat.max():.4f} "
        f"|x|<0.01={(flat.abs() < 0.01).float().mean():.1%}"
    )


def load_model(ckpt_path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def load_record(manifest_path, poem_id=None, feat_dir=FEAT_DIR):
    records = [json.loads(l) for l in open(manifest_path, encoding="utf-8")]
    if poem_id:
        rec = next(r for r in records if r["poem_id"] == poem_id)
    else:
        rec = records[0]
    npz = np.load(str(feat_dir / f"{rec['poem_id']}.npz"))
    return rec, npz


def trace_sample(model, rec, npz, label):
    print(f"\n{'='*70}")
    print(f"  [{label}] poem_id={rec['poem_id']}: {rec['text'][:40]}")
    print(f"{'='*70}")

    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
    dur_gt = torch.tensor([rec["durations"]], dtype=torch.long)
    mel_gt = torch.tensor(npz["mel"].astype(np.float32)).unsqueeze(0)
    f0_gt = torch.tensor(npz["f0"].astype(np.float32)).unsqueeze(0)
    e_gt = torch.tensor(npz["energy"].astype(np.float32)).unsqueeze(0)

    n_phones = len(rec["phoneme_ids"])
    dur_gt_total = int(dur_gt.sum())
    print(f"  Phonemes: {n_phones}, GT dur total: {dur_gt_total}, mel frames: {mel_gt.shape[1]}")
    print(f"  Phoneme IDs: {rec['phoneme_ids']}")

    with torch.no_grad():
        # ── 1. Embedding ──
        emb = model.embedding(phone_ids) * math.sqrt(model.d_model)
        print(f"\n  1. Embedding: {fmt_tensor(emb, '')}")

        # ── 2. Positional Encoding ──
        x = model.pos_enc(emb)
        pe_contribution = (x - emb)
        print(f"  2. After PosEnc: {fmt_tensor(x, '')}")
        print(f"     PE contribution: {fmt_tensor(pe_contribution, '')}")

        # ── 3. Encoder layers ──
        for i, layer in enumerate(model.encoder_layers):
            x_pre = x.clone()
            x = layer(x)
            delta = (x - x_pre).flatten().float()
            print(f"  3.{i} Encoder layer {i}: {fmt_tensor(x, '')} delta_mean={delta.abs().mean():.4f}")

        enc_out = x

        # ── 4. Duration predictor ──
        log_dur_pred = model.duration_predictor(x)
        dur_pred = log_dur_pred.exp().round().clamp(min=0).long()
        dur_pred_total = int(dur_pred.sum())

        print(f"\n  4. Duration predictor:")
        print(f"     log_dur_pred: {fmt_tensor(log_dur_pred, '')}")
        print(f"     GT durations:     {dur_gt[0].tolist()}")
        print(f"     Pred durations:   {dur_pred[0].tolist()}")
        print(f"     GT total={dur_gt_total}, Pred total={dur_pred_total}, ratio={dur_pred_total/dur_gt_total:.2f}")

        # Per-phone duration comparison
        dur_gt_flat = dur_gt[0].float()
        dur_pred_flat = dur_pred[0].float()
        dur_diff = (dur_pred_flat - dur_gt_flat)
        print(f"     Dur diff per phone: mean={dur_diff.mean():.1f} std={dur_diff.std():.1f} "
              f"min={dur_diff.min():.0f} max={dur_diff.max():.0f}")

        # ── 5. Pitch predictor ──
        pitch_pred = model.pitch_predictor(x)
        f0_norm_gt = torch.where(
            f0_gt > 0,
            (torch.log(f0_gt.clamp(min=1)) - stats["f0_mean"]) / stats["f0_std"],
            torch.zeros_like(f0_gt),
        )
        print(f"\n  5. Pitch predictor:")
        print(f"     pred: {fmt_tensor(pitch_pred, '')}")
        print(f"     GT (first {min(n_phones, 20)} phones): {f0_norm_gt[0, :n_phones].tolist()[:20]}")
        print(f"     Pred (first {min(n_phones, 20)} phones): {pitch_pred[0, :n_phones].tolist()[:20]}")

        # ── 6. Energy predictor ──
        energy_pred = model.energy_predictor(x)
        e_norm_gt = (e_gt - stats["energy_mean"]) / stats["energy_std"]
        print(f"\n  6. Energy predictor:")
        print(f"     pred: {fmt_tensor(energy_pred, '')}")
        print(f"     GT (first {min(n_phones, 20)} phones): {e_norm_gt[0, :n_phones].tolist()[:20]}")
        print(f"     Pred (first {min(n_phones, 20)} phones): {energy_pred[0, :n_phones].tolist()[:20]}")

        # ── 7. Length regulator (GT durations) ──
        mel_input_gt = model.length_regulator(enc_out, dur_gt)
        print(f"\n  7. Length regulator (GT dur): {fmt_tensor(mel_input_gt, '')}")

        # ── 8. Length regulator (predicted durations) ──
        mel_input_pred = model.length_regulator(enc_out, dur_pred)
        print(f"     Length regulator (Pred dur): {fmt_tensor(mel_input_pred, '')}")

        # ── 9. Full forward with GT variance ──
        mel_norm_gt_ref = (mel_gt - torch.tensor(stats["mel_mean"])) / torch.tensor(stats["mel_std"])
        # GT-var path (manual, like eval scripts)
        x_gt = enc_out.clone()
        mi = model.length_regulator(x_gt, dur_gt)
        T_out = mi.size(1)
        pitch_gt_expanded = f0_norm_gt[:, :T_out]
        energy_gt_expanded = e_norm_gt[:, :T_out]
        mi = mi + model.pitch_embed(pitch_gt_expanded.unsqueeze(-1)) + \
                 model.energy_embed(energy_gt_expanded.unsqueeze(-1))
        mi = model.pos_enc(mi)
        dec = mi
        for layer in model.decoder_layers:
            dec = layer(dec)
        mel_gtvar = model.mel_linear(dec)

        print(f"\n  9. Decoder output (GT variance):")
        print(f"     mel_pred_norm: {fmt_tensor(mel_gtvar, '')}")
        T = min(mel_gtvar.size(1), mel_norm_gt_ref.size(1))
        l1_gt = (mel_gtvar[:, :T] - mel_norm_gt_ref[:, :T]).abs().mean().item()
        print(f"     Mel L1 vs GT (GT-var): {l1_gt:.4f}")

        # ── 10. Full forward with predicted variance ──
        mel_predvar, _, _, _ = model.forward(phone_ids)
        print(f"\n 10. Decoder output (Pred variance):")
        print(f"     mel_pred_norm: {fmt_tensor(mel_predvar, '')}")
        T2 = mel_predvar.size(1)
        T2_cmp = min(T2, mel_norm_gt_ref.size(1))
        l1_pred = (mel_predvar[:, :T2_cmp] - mel_norm_gt_ref[:, :T2_cmp]).abs().mean().item()
        print(f"     Mel L1 vs GT (Pred-var, truncated): {l1_pred:.4f}")

        # ── 11. Mel analysis ──
        mel_gtvar_raw = mel_gtvar[0].numpy() * np.array(stats["mel_std"]) + np.array(stats["mel_mean"])
        mel_predvar_raw = mel_predvar[0].numpy() * np.array(stats["mel_std"]) + np.array(stats["mel_mean"])
        mel_gt_raw = mel_gt[0].numpy()

        print(f"\n 11. Mel spectrogram analysis:")
        print(f"     GT mel:      mean={mel_gt_raw.mean():.2f} std={mel_gt_raw.std():.2f} "
              f"min={mel_gt_raw.min():.2f} max={mel_gt_raw.max():.2f}")
        print(f"     GT-var mel:  mean={mel_gtvar_raw.mean():.2f} std={mel_gtvar_raw.std():.2f} "
              f"min={mel_gtvar_raw.min():.2f} max={mel_gtvar_raw.max():.2f}")
        print(f"     Pred-var mel:mean={mel_predvar_raw.mean():.2f} std={mel_predvar_raw.std():.2f} "
              f"min={mel_predvar_raw.min():.2f} max={mel_predvar_raw.max():.2f}")

        # Frame energy distribution (are some frames silent?)
        gt_frame_energy = mel_gt_raw.mean(axis=1)
        gtvar_frame_energy = mel_gtvar_raw.mean(axis=1)
        predvar_frame_energy = mel_predvar_raw.mean(axis=1)
        print(f"\n     GT mel frame energy:      "
              f"mean={gt_frame_energy.mean():.2f} silent(<-5)={np.sum(gt_frame_energy < -5)}/{len(gt_frame_energy)}")
        print(f"     GT-var mel frame energy:  "
              f"mean={gtvar_frame_energy.mean():.2f} silent(<-5)={np.sum(gtvar_frame_energy < -5)}/{len(gtvar_frame_energy)}")
        print(f"     Pred-var mel frame energy:"
              f"mean={predvar_frame_energy.mean():.2f} silent(<-5)={np.sum(predvar_frame_energy < -5)}/{len(predvar_frame_energy)}")


def check_phoneme_coverage():
    """Check if holdout poems use phoneme IDs not seen in training."""
    print(f"\n{'='*70}")
    print("  Phoneme ID coverage analysis")
    print(f"{'='*70}")

    train_ids = set()
    train_counts = Counter()
    for rec in [json.loads(l) for l in open(TRAIN_MANIFEST, encoding="utf-8")]:
        for pid in rec["phoneme_ids"]:
            train_ids.add(pid)
            train_counts[pid] += 1

    holdout_ids = set()
    holdout_missing = []
    for rec in [json.loads(l) for l in open(HOLDOUT_MANIFEST, encoding="utf-8")]:
        for pid in rec["phoneme_ids"]:
            holdout_ids.add(pid)
            if pid not in train_ids:
                holdout_missing.append((pid, rec["poem_id"]))

    print(f"  Train: {len(train_ids)} unique phoneme IDs")
    print(f"  Holdout: {len(holdout_ids)} unique phoneme IDs")
    print(f"  Holdout IDs NOT in training: {len(holdout_missing)}")
    if holdout_missing:
        for pid, poid in holdout_missing[:20]:
            print(f"    pid={pid} in {poid}")
    else:
        print(f"    (none) - all holdout phonemes seen in training")

    # Check embedding weights for unseen IDs
    return train_ids


def check_embeddings(model, train_ids):
    """Check if embedding weights for train-only vs all IDs differ significantly."""
    print(f"\n{'='*70}")
    print("  Embedding weight analysis")
    print(f"{'='*70}")

    emb = model.embedding.weight.data  # [vocab, d_model]
    all_ids = set(range(emb.shape[0]))
    padding_idx = 0

    # Skip padding idx
    train_only = sorted(train_ids)
    unseen = sorted(all_ids - train_ids - {padding_idx})

    if unseen:
        emb_train = emb[train_only]
        emb_unseen = emb[unseen]
        print(f"  Train IDs embeddings: mean={emb_train.mean():.4f} std={emb_train.std():.4f}")
        print(f"  Unseen IDs embeddings: mean={emb_unseen.mean():.4f} std={emb_unseen.std():.4f}")
    else:
        print(f"  No unseen IDs (all vocab IDs are in training)")

    # Check if padding embedding leaked into other positions
    print(f"  Padding (0) embedding norm: {emb[0].norm():.4f}")
    print(f"  Mean embedding norm: {emb.norm(dim=1).mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=str(ROOT / "checkpoints" / "D300e2e_step24000_slim.pt"),
    )
    parser.add_argument("--train_poem_id", default=None, help="Specific training poem to trace")
    parser.add_argument("--holdout_poem_id", default="poem_0000", help="Specific holdout poem to trace")
    a = parser.parse_args()

    print("Loading model...")
    model, ckpt = load_model(a.ckpt)
    print(f"Checkpoint step: {ckpt.get('step', '?')}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,} ({n_params/1e6:.1f}M)")

    # Phoneme coverage
    train_ids = check_phoneme_coverage()
    check_embeddings(model, train_ids)

    # Trace a training sample (pick one from train_300)
    train_rec, train_npz = load_record(TRAIN_MANIFEST, a.train_poem_id)
    trace_sample(model, train_rec, train_npz, "TRAIN")

    # Trace a holdout sample
    holdout_rec, holdout_npz = load_record(HOLDOUT_MANIFEST, a.holdout_poem_id)
    trace_sample(model, holdout_rec, holdout_npz, "HOLDOUT")

    print(f"\n{'='*70}")
    print("Diagnostic complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
