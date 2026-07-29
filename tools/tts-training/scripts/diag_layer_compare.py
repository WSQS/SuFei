"""Layer-by-layer comparison: training forward vs A0 inference.

For the same sample, runs both paths and compares every intermediate tensor.
Also verifies pitch/energy normalization space consistency.

Training path (B' mode, nar_train.py --distill --gt_variance):
  - encoder with phone_mask
  - standalone length_regulate_batch
  - GT pitch/energy (normalized)
  - decoder without mask

A0 path (eval scripts):
  - encoder without mask
  - model.length_regulator
  - GT pitch/energy (normalized)
  - decoder without mask
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


def compare(name, a, b):
    if a.shape != b.shape:
        print(f"  {name:30s} SHAPE MISMATCH: {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    diff = (a - b).abs()
    print(f"  {name:30s} max={diff.max().item():.8f}  mean={diff.mean().item():.8f}  "
          f"shape={tuple(a.shape)}")


def main():
    with open(ROOT / "data" / "paddle_distill_norm_stats.json") as f:
        stats = json.load(f)

    mel_mean = torch.tensor(stats["mel_mean"], dtype=torch.float32)
    mel_std = torch.tensor(stats["mel_std"], dtype=torch.float32)
    f0_mean = stats["f0_mean"]
    f0_std = stats["f0_std"]
    energy_mean = stats["energy_mean"]
    energy_std = stats["energy_std"]

    ckpt = torch.load(str(ROOT / "checkpoints" / "fs2_bprime_8overfit.pt"),
                      map_location=device, weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    manifest = [json.loads(l) for l in open(ROOT / "data" / "paddle_distill_manifest.jsonl", encoding="utf-8")]
    manifest.sort(key=lambda r: r["mel_len"])

    lr_standalone = LengthRegulator()

    for target_id in ["poem_0244", "poem_0097", "poem_0298"]:
        rec = next(r for r in manifest if r["poem_id"] == target_id)
        npz = np.load(str(ROOT / "data" / "paddle_distill_features" / f"{target_id}.npz"))

        mel_gt = torch.tensor(npz["mel"].astype(np.float32)).unsqueeze(0)
        f0_gt = torch.tensor(npz["f0"].astype(np.float32)).unsqueeze(0)
        e_gt = torch.tensor(npz["energy"].astype(np.float32)).unsqueeze(0)
        dur_gt = torch.tensor([rec["durations"]], dtype=torch.long)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
        phone_mask = torch.zeros(1, len(rec["phoneme_ids"]), dtype=torch.bool)

        # ── Normalization (shared) ──
        mel_norm = (mel_gt - mel_mean) / mel_std
        f0_norm = torch.where(
            f0_gt > 0,
            (torch.log(f0_gt.clamp(min=1)) - f0_mean) / f0_std,
            torch.zeros_like(f0_gt),
        )
        e_norm = (e_gt - energy_mean) / energy_std

        print(f"\n{'='*70}")
        print(f"Sample: {target_id}  (n_phones={len(rec['phoneme_ids'])}, mel_len={rec['mel_len']})")
        print(f"{'='*70}")

        # ── Pitch/Energy normalization check ──
        print(f"\n  Pitch/Energy normalization check:")
        print(f"    f0_norm:    mean={f0_norm.mean():.4f}, std={f0_norm.std():.4f}, range=[{f0_norm.min():.4f}, {f0_norm.max():.4f}]")
        print(f"    e_norm:     mean={e_norm.mean():.4f}, std={e_norm.std():.4f}, range=[{e_norm.min():.4f}, {e_norm.max():.4f}]")

        # ════════════════════════════════════════════════��═══════════════════
        # PATH A: Exact training forward (nar_train.py --distill --gt_variance)
        # ════════════════════════════════════════════════════════════════════
        with torch.no_grad():
            # Encoder
            x_train = model.embedding(phone_ids) * math.sqrt(model.d_model)
            x_train = model.pos_enc(x_train)
            for layer in model.encoder_layers:
                x_train = layer(x_train, mask=phone_mask)

            # Predictors (detached in gt_variance mode)
            log_dur_pred = model.duration_predictor(x_train).detach()
            pitch_pred_enc = model.pitch_predictor(x_train).detach()
            energy_pred_enc = model.energy_predictor(x_train).detach()

            # Length regulate with GT durations (standalone)
            mel_input_train = nar_train.length_regulate_batch(x_train, dur_gt)
            T_pred_train = mel_input_train.size(1)

            # GT pitch/energy
            T_var = min(T_pred_train, mel_gt.size(1))
            pitch_expanded_train = f0_norm[:, :T_var]
            energy_expanded_train = e_norm[:, :T_var]

            pitch_embed_train = model.pitch_embed(pitch_expanded_train[:, :T_pred_train].unsqueeze(-1))
            energy_embed_train = model.energy_embed(energy_expanded_train[:, :T_pred_train].unsqueeze(-1))

            mel_input_train = mel_input_train + pitch_embed_train + energy_embed_train
            mel_input_train = model.pos_enc(mel_input_train)

            dec_train = mel_input_train
            for layer in model.decoder_layers:
                dec_train = layer(dec_train)
            mel_pred_train = model.mel_linear(dec_train)

        # ════════════════════════════════════════════════════════════════════
        # PATH B: Exact A0 inference (eval_8overfit.py / bprime_a0_validate.py)
        # ════════════════════════════════════════════════════════════════════
        with torch.no_grad():
            x_a0 = model.embedding(phone_ids) * math.sqrt(model.d_model)
            x_a0 = model.pos_enc(x_a0)
            for layer in model.encoder_layers:
                x_a0 = layer(x_a0)  # NO mask

            mel_input_a0 = model.length_regulator(x_a0, dur_gt)
            T_out_a0 = mel_input_a0.size(1)

            # GT pitch/energy (same normalization)
            pitch_embed_a0 = model.pitch_embed(f0_norm[:, :T_out_a0].unsqueeze(-1))
            energy_embed_a0 = model.energy_embed(e_norm[:, :T_out_a0].unsqueeze(-1))

            mel_input_a0 = mel_input_a0 + pitch_embed_a0 + energy_embed_a0
            mel_input_a0 = model.pos_enc(mel_input_a0)

            dec_a0 = mel_input_a0
            for layer in model.decoder_layers:
                dec_a0 = layer(dec_a0)
            mel_pred_a0 = model.mel_linear(dec_a0)

        # ════════════════════════════════════════════════════════════════════
        # COMPARISON
        # ════════════════════════════════════════════════════════════════════

        print(f"\n  Layer-by-layer comparison (train vs A0):")

        # Re-run both paths capturing intermediates
        with torch.no_grad():
            # ── Embedding ──
            emb_t = model.embedding(phone_ids) * math.sqrt(model.d_model)
            emb_t = model.pos_enc(emb_t)
            emb_a = model.embedding(phone_ids) * math.sqrt(model.d_model)
            emb_a = model.pos_enc(emb_a)
            compare("after_pos_enc", emb_t, emb_a)

            # ── Encoder layers ──
            enc_t = emb_t
            enc_a = emb_a
            for i, layer in enumerate(model.encoder_layers):
                enc_t = layer(enc_t, mask=phone_mask)
                enc_a = layer(enc_a)
                compare(f"encoder_layer_{i}", enc_t, enc_a)

            # ── Length regulator ──
            lr_t = nar_train.length_regulate_batch(enc_t, dur_gt)
            lr_a = model.length_regulator(enc_a, dur_gt)
            compare(f"length_regulated", lr_t, lr_a)

            # ── Pitch/energy expansion ──
            T_t = lr_t.size(1)
            T_a = lr_a.size(1)
            print(f"    T_train={T_t}, T_a0={T_a}")

            T_var_t = min(T_t, mel_gt.size(1))
            pitch_t = f0_norm[:, :T_var_t]
            energy_t = e_norm[:, :T_var_t]

            T_var_a = min(T_a, mel_gt.size(1))
            pitch_a = f0_norm[:, :T_var_a]
            energy_a = e_norm[:, :T_var_a]

            pe_t = model.pitch_embed(pitch_t[:, :T_t].unsqueeze(-1))
            pe_a = model.pitch_embed(pitch_a[:, :T_a].unsqueeze(-1))
            compare("pitch_embed", pe_t, pe_a)

            ee_t = model.energy_embed(energy_t[:, :T_t].unsqueeze(-1))
            ee_a = model.energy_embed(energy_a[:, :T_a].unsqueeze(-1))
            compare("energy_embed", ee_t, ee_a)

            # ── Decoder input ──
            dec_in_t = lr_t + pe_t + ee_t
            dec_in_a = lr_a + pe_a + ee_a
            dec_in_t = model.pos_enc(dec_in_t)
            dec_in_a = model.pos_enc(dec_in_a)
            compare("decoder_input", dec_in_t, dec_in_a)

            # ── Decoder layers ──
            d_t = dec_in_t
            d_a = dec_in_a
            for i, layer in enumerate(model.decoder_layers):
                d_t = layer(d_t)
                d_a = layer(d_a)
                compare(f"decoder_layer_{i}", d_t, d_a)

            # ── Mel output ──
            mel_t = model.mel_linear(d_t)
            mel_a = model.mel_linear(d_a)
            compare("mel_output (normalized)", mel_t, mel_a)

            # ── Mel L1 vs target ──
            T_min = min(mel_t.size(1), mel_norm.size(1), mel_a.size(1))
            l1_train = (mel_t[:, :T_min] - mel_norm[:, :T_min]).abs().mean()
            l1_a0 = (mel_a[:, :T_min] - mel_norm[:, :T_min]).abs().mean()
            print(f"\n  Mel L1 vs GT (normalized space):")
            print(f"    Train path: {l1_train.item():.6f}")
            print(f"    A0 path:    {l1_a0.item():.6f}")

            # ── Also check duration sum vs mel_len ──
            dur_sum = int(dur_gt.sum().item())
            mel_len = rec["mel_len"]
            print(f"\n  Duration check: sum(dur)={dur_sum}, mel_len={mel_len}, diff={dur_sum - mel_len}")


if __name__ == "__main__":
    main()
