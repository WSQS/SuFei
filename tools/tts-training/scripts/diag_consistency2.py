"""Diagnostic 2: Precise length regulator comparison + training/A0 mel diff.

Key finding from diag_consistency.py:
1. A training uses PREDICTED pitch/energy, A0 uses GT → mismatch
2. length_regulate_batch vs LengthRegulator differ on batch data

This script checks:
- Do the two length regulators differ for SINGLE samples (batch=1)?
- What is the EXACT mel diff for B' (training path vs A0 path)?
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2, LengthRegulator
import nar_train

device = "cpu"


def test_length_regulators_single():
    """Test if length regulators agree for single samples (batch=1)."""
    print("=" * 70)
    print("TEST 1: Length Regulator Consistency (single sample, batch=1)")
    print("=" * 70)

    # Test with real duration data
    manifest = [json.loads(l) for l in open(ROOT / "data" / "train_manifest.jsonl", encoding="utf-8")]
    manifest.sort(key=lambda r: r["mel_len"])

    lr_model = LengthRegulator()
    n_mismatch = 0

    for rec in manifest[:20]:
        dur = torch.tensor([rec["durations"]], dtype=torch.long)
        dummy_x = torch.randn(1, len(rec["durations"]), 256)

        out_model = lr_model(dummy_x, dur)
        out_standalone = nar_train.length_regulate_batch(dummy_x, dur)

        if out_model.shape != out_standalone.shape:
            print(f"  {rec['poem_id']}: SHAPE MISMATCH model={out_model.shape} vs standalone={out_standalone.shape}")
            n_mismatch += 1
            continue

        diff = (out_model - out_standalone).abs()
        max_d = diff.max().item()
        if max_d > 1e-5:
            print(f"  {rec['poem_id']}: MISMATCH max_diff={max_d:.6f} mean_diff={diff.mean().item():.6f}")
            # Find where they differ
            diff_per_frame = diff.mean(dim=-1).squeeze(0)  # [T]
            bad_frames = (diff_per_frame > 1e-5).nonzero(as_tuple=True)[0]
            if len(bad_frames) > 0:
                print(f"    First bad frame: {bad_frames[0].item()}, total bad: {len(bad_frames)}/{len(diff_per_frame)}")
                # Check the phoneme assignments
                T = out_model.size(1)
                frame_idx = torch.arange(T)
                cumdurs = dur.cumsum(dim=1).squeeze(0)
                starts = cumdurs - dur.squeeze(0)
                d = dur.squeeze(0)
                boundaries = starts + d

                idx_bucket = torch.bucketize(frame_idx, boundaries, right=False).clamp(max=len(d) - 1)

                # Model's assignment
                frame_indices = torch.arange(T).unsqueeze(0)
                phoneme_per_frame = (cumdurs.unsqueeze(0).unsqueeze(2) > frame_indices.unsqueeze(1)).float().argmax(dim=1).squeeze(0)
                phoneme_per_frame = phoneme_per_frame.clamp(max=len(d) - 1)

                diffs_idx = (idx_bucket != phoneme_per_frame)
                if diffs_idx.any():
                    first_diff = diffs_idx.nonzero()[0].item()
                    print(f"    Frame {first_diff}: bucketize→phoneme {idx_bucket[first_diff].item()}, model→phoneme {phoneme_per_frame[first_diff].item()}")
                    print(f"    Durations: {d.tolist()}")
                    print(f"    Cumdurs:   {cumdurs.tolist()}")
                    print(f"    Boundaries (starts+d): {boundaries.tolist()}")
            n_mismatch += 1
        else:
            pass  # identical

    if n_mismatch == 0:
        print("  All 20 samples: IDENTICAL (< 1e-5)")
    else:
        print(f"\n  {n_mismatch}/20 samples have mismatches")


def test_length_regulators_padded_batch():
    """Test with padded batch to understand padding behavior."""
    print(f"\n{'='*70}")
    print("TEST 2: Length Regulator Consistency (padded batch=2)")
    print("=" * 70)

    dur1 = torch.tensor([[3, 2, 1]], dtype=torch.long)  # total=6
    dur2 = torch.tensor([[1, 1, 1]], dtype=torch.long)  # total=3
    dur_batch = torch.cat([dur1, dur2], dim=0)  # [2, 3]
    dummy_x = torch.randn(2, 3, 8)

    lr_model = LengthRegulator()
    out_model = lr_model(dummy_x, dur_batch)
    out_standalone = nar_train.length_regulate_batch(dummy_x, dur_batch)

    print(f"  Model output:     {out_model.shape}")
    print(f"  Standalone output: {out_standalone.shape}")

    # Sample 1 (longer)
    diff1 = (out_model[0] - out_standalone[0]).abs()
    print(f"\n  Sample 0 (dur=[3,2,1], total=6):")
    print(f"    Max diff: {diff1.max().item():.6f}")

    # Sample 2 (shorter, has padding)
    diff2 = (out_model[1] - out_standalone[1]).abs()
    print(f"\n  Sample 1 (dur=[1,1,1], total=3, T_max=6):")
    print(f"    Max diff: {diff2.max().item():.6f}")
    print(f"    Frames 3-5 (padding region):")
    print(f"      Model:     {out_model[1, 3:6, 0].tolist()}")
    print(f"      Standalone: {out_standalone[1, 3:6, 0].tolist()}")

    if diff2.max().item() > 1e-5:
        print(f"\n    *** PADDING BEHAVIOR DIFFERS ***")
        print(f"    Model assigns padding frames to last phoneme (index 2)")
        print(f"    Standalone zeros out padding frames")


def test_bprime_exact_paths():
    """Test B' with EXACT training path (standalone length_regulate_batch) vs A0."""
    print(f"\n{'='*70}")
    print("TEST 3: B' Exact Training Path vs A0 Inference Path")
    print("=" * 70)

    b_ckpt_path = ROOT / "checkpoints" / "fs2_bprime_final.pt"
    if not b_ckpt_path.exists():
        print("  B' checkpoint not found, skipping")
        return

    with open(ROOT / "data" / "paddle_distill_norm_stats.json") as f:
        stats = json.load(f)

    mel_mean = torch.tensor(stats["mel_mean"], dtype=torch.float32)
    mel_std = torch.tensor(stats["mel_std"], dtype=torch.float32)

    manifest = [json.loads(l) for l in open(ROOT / "data" / "paddle_distill_manifest.jsonl", encoding="utf-8")]
    manifest.sort(key=lambda r: r["mel_len"])

    ckpt = torch.load(str(b_ckpt_path), map_location=device, weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    lr_model = LengthRegulator()

    # Test 3 poems
    for target_id in ["poem_0244", "poem_0000", "poem_0097"]:
        rec = next((r for r in manifest if r["poem_id"] == target_id), None)
        if rec is None:
            continue
        npz = np.load(str(ROOT / "data" / "paddle_distill_features" / f"{target_id}.npz"))

        mel_gt = torch.tensor(npz["mel"].astype(np.float32)).unsqueeze(0)
        f0_gt = torch.tensor(npz["f0"].astype(np.float32)).unsqueeze(0)
        e_gt = torch.tensor(npz["energy"].astype(np.float32)).unsqueeze(0)
        dur_gt = torch.tensor([rec["durations"]], dtype=torch.long)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
        phone_mask = torch.zeros(1, len(rec["phoneme_ids"]), dtype=torch.bool)

        # Normalize
        mel_norm = (mel_gt - mel_mean) / mel_std
        f0_norm = torch.where(
            f0_gt > 0,
            (torch.log(f0_gt.clamp(min=1)) - stats["f0_mean"]) / stats["f0_std"],
            torch.zeros_like(f0_gt),
        )
        e_norm = (e_gt - torch.tensor(stats["energy_mean"])) / torch.tensor(stats["energy_std"])

        # ── Exact training path (nar_train.py with --distill --gt_variance) ──
        with torch.no_grad():
            x = model.embedding(phone_ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x, mask=phone_mask)

            # Training uses standalone length_regulate_batch
            mel_input_train = nar_train.length_regulate_batch(x, dur_gt)
            T_pred = mel_input_train.size(1)

            T_var = min(T_pred, mel_gt.size(1))
            pitch_expanded = f0_norm[:, :T_var]
            energy_expanded = e_norm[:, :T_var]

            pitch_embed = model.pitch_embed(pitch_expanded[:, :T_pred].unsqueeze(-1))
            energy_embed = model.energy_embed(energy_expanded[:, :T_pred].unsqueeze(-1))

            mel_input_train = mel_input_train + pitch_embed + energy_embed
            mel_input_train = model.pos_enc(mel_input_train)

            dec = mel_input_train
            for layer in model.decoder_layers:
                dec = layer(dec)
            mel_train = model.mel_linear(dec)

        # ── Exact A0 path (bprime_a0_validate.py) ──
        with torch.no_grad():
            x2 = model.embedding(phone_ids) * math.sqrt(model.d_model)
            x2 = model.pos_enc(x2)
            for layer in model.encoder_layers:
                x2 = layer(x2)  # NO mask
            mel_input_a0 = model.length_regulator(x2, dur_gt)  # model's LR
            T_out = mel_input_a0.size(1)
            mel_input_a0 = mel_input_a0 + \
                model.pitch_embed(f0_norm[:, :T_out].unsqueeze(-1)) + \
                model.energy_embed(e_norm[:, :T_out].unsqueeze(-1))
            mel_input_a0 = model.pos_enc(mel_input_a0)
            dec2 = mel_input_a0
            for layer in model.decoder_layers:
                dec2 = layer(dec2)
            mel_a0 = model.mel_linear(dec2)

        T_min = min(mel_train.size(1), mel_a0.size(1))
        diff = (mel_train[:, :T_min] - mel_a0[:, :T_min]).abs()

        # Also test: A0 with standalone LR (to isolate LR effect)
        with torch.no_grad():
            mel_input_a0b = nar_train.length_regulate_batch(x2, dur_gt)
            T_outb = mel_input_a0b.size(1)
            mel_input_a0b = mel_input_a0b + \
                model.pitch_embed(f0_norm[:, :T_outb].unsqueeze(-1)) + \
                model.energy_embed(e_norm[:, :T_outb].unsqueeze(-1))
            mel_input_a0b = model.pos_enc(mel_input_a0b)
            dec3 = mel_input_a0b
            for layer in model.decoder_layers:
                dec3 = layer(dec3)
            mel_a0b = model.mel_linear(dec3)

        diff_b = (mel_train[:, :T_min] - mel_a0b[:, :T_min]).abs()

        # And test: LR diff on this specific input
        lr_diff = (nar_train.length_regulate_batch(x, dur_gt) - model.length_regulator(x, dur_gt)).abs()

        print(f"\n  {target_id}:")
        print(f"    LR diff (standalone vs model): max={lr_diff.max().item():.6f}")
        print(f"    Mel diff (train vs A0):        max={diff.max().item():.6f} mean={diff.mean().item():.6f}")
        print(f"    Mel diff (train vs A0+standalone LR): max={diff_b.max().item():.6f} mean={diff_b.mean().item():.6f}")


def test_a_exact_paths():
    """Test A with EXACT training path (predicted pitch/energy) vs A0 (GT pitch/energy)."""
    print(f"\n{'='*70}")
    print("TEST 4: A Exact Training Path vs A0 Inference Path")
    print("=" * 70)

    a_ckpt_path = ROOT / "checkpoints" / "acoustic_baseline_step12000.pt"
    if not a_ckpt_path.exists():
        print("  A checkpoint not found, skipping")
        return

    with open(ROOT / "data" / "norm_stats.json") as f:
        stats = json.load(f)

    mel_mean = torch.tensor(stats["mel_mean"], dtype=torch.float32)
    mel_std = torch.tensor(stats["mel_std"], dtype=torch.float32)

    manifest = [json.loads(l) for l in open(ROOT / "data" / "train_manifest.jsonl", encoding="utf-8")]
    manifest.sort(key=lambda r: r["mel_len"])

    ckpt = torch.load(str(a_ckpt_path), map_location=device, weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for target_id in ["poem_0244", "poem_0285", "poem_0000"]:
        rec = next((r for r in manifest if r["poem_id"] == target_id), None)
        if rec is None:
            continue
        npz = np.load(str(ROOT / "data" / "nar_features" / f"{target_id}.npz"))

        mel_gt = torch.tensor(npz["mel"].astype(np.float32)).unsqueeze(0)
        f0_gt = torch.tensor(npz["f0"].astype(np.float32)).unsqueeze(0)
        e_gt = torch.tensor(npz["energy"].astype(np.float32)).unsqueeze(0)
        dur_gt = torch.tensor([rec["durations"]], dtype=torch.long)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)

        mel_norm = (mel_gt - mel_mean) / mel_std
        f0_norm = torch.where(
            f0_gt > 0,
            (torch.log(f0_gt.clamp(min=1)) - stats["f0_mean"]) / stats["f0_std"],
            torch.zeros_like(f0_gt),
        )
        e_norm = (e_gt - torch.tensor(stats["energy_mean"])) / torch.tensor(stats["energy_std"])

        # ── Training path (A mode: PREDICTED pitch/energy) ──
        with torch.no_grad():
            x = model.embedding(phone_ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x)
            pitch_pred = model.pitch_predictor(x)
            energy_pred = model.energy_predictor(x)

            mel_input_train = nar_train.length_regulate_batch(x, dur_gt)
            T_pred = mel_input_train.size(1)

            pitch_expanded = nar_train.length_regulate_batch(pitch_pred.unsqueeze(-1), dur_gt).squeeze(-1)
            energy_expanded = nar_train.length_regulate_batch(energy_pred.unsqueeze(-1), dur_gt).squeeze(-1)

            pitch_embed = model.pitch_embed(pitch_expanded[:, :T_pred].unsqueeze(-1))
            energy_embed = model.energy_embed(energy_expanded[:, :T_pred].unsqueeze(-1))

            mel_input_train = mel_input_train + pitch_embed + energy_embed
            mel_input_train = model.pos_enc(mel_input_train)
            dec = mel_input_train
            for layer in model.decoder_layers:
                dec = layer(dec)
            mel_train = model.mel_linear(dec)

        # ── A0 path (GT pitch/energy) ──
        with torch.no_grad():
            x2 = model.embedding(phone_ids) * math.sqrt(model.d_model)
            x2 = model.pos_enc(x2)
            for layer in model.encoder_layers:
                x2 = layer(x2)
            mel_input_a0 = model.length_regulator(x2, dur_gt)
            T_out = mel_input_a0.size(1)
            mel_input_a0 = mel_input_a0 + \
                model.pitch_embed(f0_norm[:, :T_out].unsqueeze(-1)) + \
                model.energy_embed(e_norm[:, :T_out].unsqueeze(-1))
            mel_input_a0 = model.pos_enc(mel_input_a0)
            dec2 = mel_input_a0
            for layer in model.decoder_layers:
                dec2 = layer(dec2)
            mel_a0 = model.mel_linear(dec2)

        # ── A0 with predicted pitch/energy (matching training) ──
        with torch.no_grad():
            mel_input_a0p = model.length_regulator(x2, dur_gt)
            T_outp = mel_input_a0p.size(1)
            pitch_exp2 = model.length_regulator(pitch_pred.unsqueeze(-1), dur_gt).squeeze(-1)
            energy_exp2 = model.length_regulator(energy_pred.unsqueeze(-1), dur_gt).squeeze(-1)
            mel_input_a0p = mel_input_a0p + \
                model.pitch_embed(pitch_exp2[:, :T_outp].unsqueeze(-1)) + \
                model.energy_embed(energy_exp2[:, :T_outp].unsqueeze(-1))
            mel_input_a0p = model.pos_enc(mel_input_a0p)
            dec3 = mel_input_a0p
            for layer in model.decoder_layers:
                dec3 = layer(dec3)
            mel_a0p = model.mel_linear(dec3)

        T_min = min(mel_train.size(1), mel_a0.size(1))
        diff_gt = (mel_train[:, :T_min] - mel_a0[:, :T_min]).abs()
        diff_pred = (mel_train[:, :T_min] - mel_a0p[:, :T_min]).abs()

        # Also compare GT pitch vs predicted pitch
        T_p = min(f0_norm.size(1), pitch_expanded.size(1))
        pitch_diff = (f0_norm[:, :T_p] - pitch_expanded[:, :T_p]).abs()

        print(f"\n  {target_id}:")
        print(f"    Mel diff (train vs A0-GT):     max={diff_gt.max().item():.4f} mean={diff_gt.mean().item():.4f}")
        print(f"    Mel diff (train vs A0-pred):   max={diff_pred.max().item():.4f} mean={diff_pred.mean().item():.4f}")
        print(f"    Pitch diff (GT vs predicted):  mean={pitch_diff.mean().item():.4f} max={pitch_diff.max().item():.4f}")
        print(f"    Pitch pred range: [{pitch_pred.min():.3f}, {pitch_pred.max():.3f}]")
        print(f"    Pitch GT range:   [{f0_norm.min():.3f}, {f0_norm.max():.3f}]")


if __name__ == "__main__":
    test_length_regulators_single()
    test_length_regulators_padded_batch()
    test_bprime_exact_paths()
    test_a_exact_paths()
