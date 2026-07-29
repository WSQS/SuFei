"""Diagnostic: check if training forward path and A0 inference path produce
identical mel output for the same input.

This is the #1 risk: if training and A0 inference use different forward
paths, all CER evaluations are misleading.

Key differences to check:
1. Training: encoder layers get phone_mask; A0: no mask
2. Training: uses standalone length_regulate_batch(); A0: uses model.length_regulator()
3. Training (A, no --gt_variance): decoder input uses PREDICTED pitch/energy
   A0 inference: decoder input uses GT pitch/energy → MISMATCH for A experiment
4. Training (B', --gt_variance): decoder input uses GT pitch/energy → matches A0
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

device = "cpu"


def training_forward_a(model, phoneme_ids, durations_gt, mel_gt, f0_gt, energy_gt,
                       phone_mask, mel_mean, mel_std, f0_mean, f0_std,
                       energy_mean, energy_std):
    """Replicate the training forward path for experiment A (no --gt_variance)."""
    # Normalize
    mel_norm = (mel_gt - mel_mean) / mel_std
    f0_norm = torch.where(
        f0_gt > 0,
        (torch.log(f0_gt.clamp(min=1)) - f0_mean) / f0_std,
        torch.zeros_like(f0_gt),
    )
    energy_norm = (energy_gt - energy_mean) / energy_std

    x = model.embedding(phoneme_ids) * math.sqrt(model.d_model)
    x = model.pos_enc(x)
    for layer in model.encoder_layers:
        x = layer(x, mask=phone_mask)

    log_dur_pred = model.duration_predictor(x)
    pitch_pred_enc = model.pitch_predictor(x)
    energy_pred_enc = model.energy_predictor(x)

    # Use model's length regulator (same as training standalone for comparison)
    lr = LengthRegulator()
    mel_input = lr(x, durations_gt)
    T_pred = mel_input.size(1)

    # A mode: PREDICTED pitch/energy expanded
    pitch_expanded = lr(pitch_pred_enc.unsqueeze(-1), durations_gt).squeeze(-1)
    energy_expanded = lr(energy_pred_enc.unsqueeze(-1), durations_gt).squeeze(-1)

    pitch_embed = model.pitch_embed(pitch_expanded[:, :T_pred].unsqueeze(-1))
    energy_embed = model.energy_embed(energy_expanded[:, :T_pred].unsqueeze(-1))

    mel_input = mel_input + pitch_embed + energy_embed
    mel_input = model.pos_enc(mel_input)

    dec = mel_input
    for layer in model.decoder_layers:
        dec = layer(dec)
    mel_pred = model.mel_linear(dec)

    return mel_pred, mel_norm, pitch_pred_enc, energy_pred_enc


def training_forward_bprime(model, phoneme_ids, durations_gt, mel_gt, f0_gt, energy_gt,
                            phone_mask, mel_mean, mel_std, f0_mean, f0_std,
                            energy_mean, energy_std):
    """Replicate the training forward path for experiment B' (--gt_variance)."""
    mel_norm = (mel_gt - mel_mean) / mel_std
    f0_norm = torch.where(
        f0_gt > 0,
        (torch.log(f0_gt.clamp(min=1)) - f0_mean) / f0_std,
        torch.zeros_like(f0_gt),
    )
    energy_norm = (energy_gt - energy_mean) / energy_std

    x = model.embedding(phoneme_ids) * math.sqrt(model.d_model)
    x = model.pos_enc(x)
    for layer in model.encoder_layers:
        x = layer(x, mask=phone_mask)

    log_dur_pred = model.duration_predictor(x).detach()
    pitch_pred_enc = model.pitch_predictor(x).detach()
    energy_pred_enc = model.energy_predictor(x).detach()

    lr = LengthRegulator()
    mel_input = lr(x, durations_gt)
    T_pred = mel_input.size(1)

    # B' mode: GT pitch/energy
    T_var = min(T_pred, mel_gt.size(1))
    pitch_expanded = f0_norm[:, :T_var]
    energy_expanded = energy_norm[:, :T_var]

    pitch_embed = model.pitch_embed(pitch_expanded[:, :T_pred].unsqueeze(-1))
    energy_embed = model.energy_embed(energy_expanded[:, :T_pred].unsqueeze(-1))

    mel_input = mel_input + pitch_embed + energy_embed
    mel_input = model.pos_enc(mel_input)

    dec = mel_input
    for layer in model.decoder_layers:
        dec = layer(dec)
    mel_pred = model.mel_linear(dec)

    return mel_pred, mel_norm, pitch_pred_enc, energy_pred_enc


def a0_inference(model, phoneme_ids, dur_gt_t, pitch_gt, energy_gt_t):
    """The A0 inference path used in bprime_a0_validate.py / seen_eval.py."""
    with torch.no_grad():
        x = model.embedding(phoneme_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)  # NO mask
        mel_input = model.length_regulator(x, dur_gt_t)
        T_out = mel_input.size(1)
        mel_input = mel_input + \
            model.pitch_embed(pitch_gt[:, :T_out].unsqueeze(-1)) + \
            model.energy_embed(energy_gt_t[:, :T_out].unsqueeze(-1))
        mel_input = model.pos_enc(mel_input)
        dec = mel_input
        for layer in model.decoder_layers:
            dec = layer(dec)
        mel = model.mel_linear(dec)
    return mel


def main():
    # Load A checkpoint and A stats
    a_ckpt_path = ROOT / "checkpoints" / "acoustic_baseline_step12000.pt"
    b_ckpt_path = ROOT / "checkpoints" / "fs2_bprime_final.pt"

    # A stats
    with open(ROOT / "data" / "norm_stats.json") as f:
        a_stats = json.load(f)
    a_mel_mean = torch.tensor(a_stats["mel_mean"], dtype=torch.float32)
    a_mel_std = torch.tensor(a_stats["mel_std"], dtype=torch.float32)

    # B' stats
    with open(ROOT / "data" / "paddle_distill_norm_stats.json") as f:
        b_stats = json.load(f)
    b_mel_mean = torch.tensor(b_stats["mel_mean"], dtype=torch.float32)
    b_mel_std = torch.tensor(b_stats["mel_std"], dtype=torch.float32)

    # Load one training sample for A
    a_manifest = [json.loads(l) for l in open(ROOT / "data" / "train_manifest.jsonl", encoding="utf-8")]
    a_manifest.sort(key=lambda r: r["mel_len"])

    # Pick poem_0244 (chuang qian ming yue guang) - known to work
    target_id = "poem_0244"
    rec_a = next(r for r in a_manifest if r["poem_id"] == target_id)
    npz_a = np.load(str(ROOT / "data" / "nar_features" / f"{target_id}.npz"))

    # Load one training sample for B'
    b_manifest = [json.loads(l) for l in open(ROOT / "data" / "paddle_distill_manifest.jsonl", encoding="utf-8")]
    rec_b = next((r for r in b_manifest if r["poem_id"] == target_id), None)

    print("=" * 70)
    print("DIAGNOSTIC: Training Forward vs A0 Inference Path Consistency")
    print("=" * 70)

    # ─── Check A experiment ───
    if a_ckpt_path.exists() and rec_b is None:
        pass  # skip B' checks

    if a_ckpt_path.exists():
        print(f"\n{'─'*70}")
        print(f"EXPERIMENT A: {target_id}")
        print(f"{'─'*70}")

        ckpt_a = torch.load(str(a_ckpt_path), map_location=device, weights_only=False)
        model_a = FastSpeech2(vocab_size=268, dropout=0.0)
        model_a.load_state_dict(ckpt_a["model"])
        model_a.eval()

        mel_gt_a = torch.tensor(npz_a["mel"].astype(np.float32)).unsqueeze(0)
        f0_gt_a = torch.tensor(npz_a["f0"].astype(np.float32)).unsqueeze(0)
        e_gt_a = torch.tensor(npz_a["energy"].astype(np.float32)).unsqueeze(0)
        dur_gt_a = torch.tensor([rec_a["durations"]], dtype=torch.long)
        phone_ids_a = torch.tensor([rec_a["phoneme_ids"]], dtype=torch.long)
        phone_mask_a = torch.zeros(1, len(rec_a["phoneme_ids"]), dtype=torch.bool)

        # Training forward (A mode: predicted pitch/energy)
        mel_train_a, _, pitch_pred_a, energy_pred_a = training_forward_a(
            model_a, phone_ids_a, dur_gt_a, mel_gt_a, f0_gt_a, e_gt_a,
            phone_mask_a, a_mel_mean, a_mel_std,
            a_stats["f0_mean"], a_stats["f0_std"],
            a_stats["energy_mean"], a_stats["energy_std"],
        )

        # A0 inference (GT pitch/energy)
        f0_norm_a = np.where(npz_a["f0"] > 0,
            (np.log(np.maximum(npz_a["f0"], 1)) - a_stats["f0_mean"]) / a_stats["f0_std"], 0.0
        ).astype(np.float32)
        e_norm_a = ((npz_a["energy"] - a_stats["energy_mean"]) / a_stats["energy_std"]).astype(np.float32)
        pitch_gt_a = torch.tensor(f0_norm_a).unsqueeze(0)
        energy_gt_a_t = torch.tensor(e_norm_a).unsqueeze(0)

        mel_a0_a = a0_inference(model_a, phone_ids_a, dur_gt_a, pitch_gt_a, energy_gt_a_t)

        # Compare
        T_min = min(mel_train_a.size(1), mel_a0_a.size(1))
        diff = (mel_train_a[:, :T_min] - mel_a0_a[:, :T_min]).abs()

        print(f"  Training mel shape: {mel_train_a.shape}")
        print(f"  A0 mel shape:        {mel_a0_a.shape}")
        print(f"  T_min:               {T_min}")
        print(f"  Max abs diff:        {diff.max().item():.6f}")
        print(f"  Mean abs diff:       {diff.mean().item():.6f}")

        # Check where they diverge
        if diff.max().item() > 1e-4:
            print(f"\n  *** MISMATCH DETECTED ***")
            print(f"  Root cause: A training uses PREDICTED pitch/energy for decoder,")
            print(f"  but A0 inference uses GT pitch/energy.")

            # How different are predicted vs GT pitch?
            lr = LengthRegulator()
            pitch_pred_expanded = lr(pitch_pred_a.unsqueeze(-1), dur_gt_a).squeeze(-1)
            energy_pred_expanded = lr(energy_pred_a.unsqueeze(-1), dur_gt_a).squeeze(-1)

            T_p = min(pitch_pred_expanded.size(1), pitch_gt_a.size(1))
            pitch_diff = (pitch_pred_expanded[:, :T_p] - pitch_gt_a[:, :T_p]).abs()
            energy_diff = (energy_pred_expanded[:, :T_p] - energy_gt_a_t[:, :T_p]).abs()

            print(f"\n  Pitch predictor vs GT:")
            print(f"    Mean abs diff: {pitch_diff.mean().item():.4f}")
            print(f"    Max abs diff:  {pitch_diff.max().item():.4f}")
            print(f"    Pred range:    [{pitch_pred_expanded.min():.2f}, {pitch_pred_expanded.max():.2f}]")
            print(f"    GT range:      [{pitch_gt_a.min():.2f}, {pitch_gt_a.max():.2f}]")

            print(f"\n  Energy predictor vs GT:")
            print(f"    Mean abs diff: {energy_diff.mean().item():.4f}")
            print(f"    Max abs diff:  {energy_diff.max().item():.4f}")

            # Now run A0 with PREDICTED pitch/energy instead of GT
            print(f"\n  Testing A0 with PREDICTED pitch/energy (matching training):")
            mel_a0_pred = a0_inference(
                model_a, phone_ids_a, dur_gt_a,
                pitch_pred_expanded, energy_pred_expanded
            )
            diff2 = (mel_train_a[:, :T_min] - mel_a0_pred[:, :T_min]).abs()
            print(f"    Max abs diff:  {diff2.max().item():.6f}")
            print(f"    Mean abs diff: {diff2.mean().item():.6f}")
        else:
            print(f"\n  Paths are numerically identical (< 1e-4)")

    # ─── Check B' experiment ───
    if b_ckpt_path.exists() and rec_b is not None:
        print(f"\n{'─'*70}")
        print(f"EXPERIMENT B': {target_id}")
        print(f"{'─'*70}")

        npz_b = np.load(str(ROOT / "data" / "paddle_distill_features" / f"{target_id}.npz"))

        ckpt_b = torch.load(str(b_ckpt_path), map_location=device, weights_only=False)
        model_b = FastSpeech2(vocab_size=268, dropout=0.0)
        model_b.load_state_dict(ckpt_b["model"])
        model_b.eval()

        mel_gt_b = torch.tensor(npz_b["mel"].astype(np.float32)).unsqueeze(0)
        f0_gt_b = torch.tensor(npz_b["f0"].astype(np.float32)).unsqueeze(0)
        e_gt_b = torch.tensor(npz_b["energy"].astype(np.float32)).unsqueeze(0)
        dur_gt_b = torch.tensor([rec_b["durations"]], dtype=torch.long)
        phone_ids_b = torch.tensor([rec_b["phoneme_ids"]], dtype=torch.long)
        phone_mask_b = torch.zeros(1, len(rec_b["phoneme_ids"]), dtype=torch.bool)

        # Training forward (B' mode: GT pitch/energy)
        mel_train_b, _, _, _ = training_forward_bprime(
            model_b, phone_ids_b, dur_gt_b, mel_gt_b, f0_gt_b, e_gt_b,
            phone_mask_b, b_mel_mean, b_mel_std,
            b_stats["f0_mean"], b_stats["f0_std"],
            b_stats["energy_mean"], b_stats["energy_std"],
        )

        # A0 inference
        f0_norm_b = np.where(npz_b["f0"] > 0,
            (np.log(np.maximum(npz_b["f0"], 1)) - b_stats["f0_mean"]) / b_stats["f0_std"], 0.0
        ).astype(np.float32)
        e_norm_b = ((npz_b["energy"] - b_stats["energy_mean"]) / b_stats["energy_std"]).astype(np.float32)
        pitch_gt_b = torch.tensor(f0_norm_b).unsqueeze(0)
        energy_gt_b_t = torch.tensor(e_norm_b).unsqueeze(0)

        mel_a0_b = a0_inference(model_b, phone_ids_b, dur_gt_b, pitch_gt_b, energy_gt_b_t)

        T_min_b = min(mel_train_b.size(1), mel_a0_b.size(1))
        diff_b = (mel_train_b[:, :T_min_b] - mel_a0_b[:, :T_min_b]).abs()

        print(f"  Training mel shape: {mel_train_b.shape}")
        print(f"  A0 mel shape:        {mel_a0_b.shape}")
        print(f"  T_min:               {T_min_b}")
        print(f"  Max abs diff:        {diff_b.max().item():.6f}")
        print(f"  Mean abs diff:       {diff_b.mean().item():.6f}")

        if diff_b.max().item() > 1e-4:
            print(f"\n  *** MISMATCH DETECTED ***")
        else:
            print(f"\n  Paths are numerically identical (< 1e-4)")

    # ─── Also check length_regulate_batch vs model.length_regulator ───
    print(f"\n{'─'*70}")
    print(f"LENGTH REGULATOR: standalone vs model method")
    print(f"{'─'*70}")

    # Need to import the standalone function
    sys.path.insert(0, str(ROOT / "scripts"))
    import nar_train
    # Create dummy data
    dummy_x = torch.randn(2, 5, 8)
    dummy_dur = torch.tensor([[3, 2, 1, 4, 2], [1, 1, 1, 1, 1]], dtype=torch.long)

    lr_model = LengthRegulator()
    out_model = lr_model(dummy_x, dummy_dur)
    out_standalone = nar_train.length_regulate_batch(dummy_x, dummy_dur)

    lr_diff = (out_model - out_standalone).abs()
    print(f"  Model LR output:    {out_model.shape}")
    print(f"  Standalone LR output: {out_standalone.shape}")
    print(f"  Max abs diff:       {lr_diff.max().item():.6f}")
    print(f"  Mean abs diff:      {lr_diff.mean().item():.6f}")
    if lr_diff.max().item() < 1e-5:
        print(f"  ✓ Length regulators are identical")
    else:
        print(f"  *** LENGTH REGULATORS DIFFER! ***")
        print(f"  This means training and A0 inference expand durations differently!")


if __name__ == "__main__":
    main()
