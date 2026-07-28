"""Analyze why decoder only works on some samples.

Check mel L1 per sample, mel range, audio RMS for all 204 samples
under GT conditions. Find what differentiates OK vs FAIL samples.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2

BASELINE = ROOT / "checkpoints" / "acoustic_baseline_step12000.pt"
device = "cpu"


def main():
    ckpt = torch.load(str(BASELINE), map_location=device, weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = ckpt["stats"]
    mel_mean = np.array(stats["mel_mean"])
    mel_std = np.array(stats["mel_std"])

    manifest = [json.loads(l) for l in open(ROOT / "data" / "train_manifest.jsonl", encoding="utf-8")]

    results = []
    for rec in manifest:
        npz = np.load(str(ROOT / rec["mel_path"]))
        mel_gt = npz["mel"].astype(np.float32)
        f0_gt = npz["f0"].astype(np.float32)
        e_gt = npz["energy"].astype(np.float32)
        dur_gt = np.array(rec["durations"], dtype=np.int32)
        ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long, device=device)
        dur_gt_t = torch.tensor(np.array([dur_gt]), dtype=torch.long, device=device)

        f0_norm = np.where(f0_gt > 0, (np.log(np.maximum(f0_gt, 1)) - stats["f0_mean"]) / stats["f0_std"], 0.0).astype(np.float32)
        e_norm = ((e_gt - stats["energy_mean"]) / stats["energy_std"]).astype(np.float32)
        pitch_gt = torch.tensor(f0_norm, device=device).unsqueeze(0)
        energy_gt = torch.tensor(e_norm, device=device).unsqueeze(0)

        with torch.no_grad():
            x = model.embedding(ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x)
            mel_input = model.length_regulator(x, dur_gt_t)
            T = mel_input.size(1)
            mel_input = mel_input + \
                model.pitch_embed(pitch_gt[:, :T].unsqueeze(-1)) + \
                model.energy_embed(energy_gt[:, :T].unsqueeze(-1))
            mel_input = model.pos_enc(mel_input)
            dec = mel_input
            for layer in model.decoder_layers:
                dec = layer(dec)
            mel = model.mel_linear(dec)

        mel_pred = mel[0].numpy()
        mel_norm_gt = (mel_gt - mel_mean) / mel_std
        T_min = min(mel_pred.shape[0], mel_norm_gt.shape[0])
        mel_l1 = np.abs(mel_pred[:T_min] - mel_norm_gt[:T_min]).mean()

        results.append({
            "poem_id": rec["poem_id"],
            "mel_len": rec["mel_len"],
            "mel_l1": mel_l1,
            "mel_pred_max": float(mel_pred.max()),
            "mel_gt_max": float(mel_norm_gt.max()),
            "n_phonemes": len(rec["phoneme_ids"]),
        })

    # Sort by mel L1
    results.sort(key=lambda r: r["mel_l1"])

    print(f"Mel L1 distribution (all {len(results)} samples):")
    l1s = [r["mel_l1"] for r in results]
    print(f"  P10: {np.percentile(l1s, 10):.3f}")
    print(f"  P25: {np.percentile(l1s, 25):.3f}")
    print(f"  P50: {np.percentile(l1s, 50):.3f}")
    print(f"  P75: {np.percentile(l1s, 75):.3f}")
    print(f"  P90: {np.percentile(l1s, 90):.3f}")

    print(f"\nBest 5 (lowest L1):")
    for r in results[:5]:
        print(f"  {r['poem_id']}: L1={r['mel_l1']:.3f} mel_len={r['mel_len']} max={r['mel_pred_max']:.2f}/{r['mel_gt_max']:.2f}")

    print(f"\nWorst 5 (highest L1):")
    for r in results[-5:]:
        print(f"  {r['poem_id']}: L1={r['mel_l1']:.3f} mel_len={r['mel_len']} max={r['mel_pred_max']:.2f}/{r['mel_gt_max']:.2f}")

    # Check correlation: mel_len vs mel_l1
    mel_lens = [r["mel_len"] for r in results]
    mel_l1s = [r["mel_l1"] for r in results]
    corr = np.corrcoef(mel_lens, mel_l1s)[0, 1]
    print(f"\nCorrelation(mel_len, mel_l1): {corr:.3f}")

    # Check mel_pred_max distribution
    pred_maxes = [r["mel_pred_max"] for r in results]
    gt_maxes = [r["mel_gt_max"] for r in results]
    print(f"\nMel max comparison:")
    print(f"  Pred max: mean={np.mean(pred_maxes):.2f} std={np.std(pred_maxes):.2f}")
    print(f"  GT max:   mean={np.mean(gt_maxes):.2f} std={np.std(gt_maxes):.2f}")
    print(f"  Ratio (pred/gt): {np.mean(pred_maxes)/np.mean(gt_maxes):.2%}")


if __name__ == "__main__":
    main()
