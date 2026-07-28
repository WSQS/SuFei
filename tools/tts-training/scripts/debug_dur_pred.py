"""Diagnose duration predictor output for a trained model."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nar_fastspeech2 import FastSpeech2
from nar_build_manifest import text_to_phonemes


def main():
    device = "cpu"

    # Load model
    ckpt_path = ROOT / "checkpoints" / "fs2_step6000.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load phone map
    phone_map = {}
    with open(ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0" / "phone_id_map.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                phone_map[parts[0]] = int(parts[1])

    # Load manifest to get GT durations
    manifest = []
    with open(ROOT / "data" / "train_manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            manifest.append(json.loads(line))

    # Compare predicted vs GT durations on training samples
    total_pred_sum = 0
    total_gt_sum = 0
    n = 0

    for rec in manifest[:20]:
        poem_id = rec["poem_id"]
        phone_ids = torch.tensor([rec["phoneme_ids"]], device=device)
        gt_durations = np.array(rec["durations"])

        with torch.no_grad():
            _, log_dur_pred, _, _ = model(phone_ids)

        pred_durations = log_dur_pred[0].exp().cpu().numpy()

        gt_sum = gt_durations.sum()
        pred_sum = pred_durations.sum()

        total_gt_sum += gt_sum
        total_pred_sum += pred_sum
        n += 1

        if n <= 5:
            print(f"\n{poem_id}:")
            print(f"  GT dur sum:   {gt_sum} ({len(gt_durations)} phones)")
            print(f"  Pred dur sum: {pred_sum:.0f} ({len(pred_durations)} phones)")
            print(f"  Ratio: {pred_sum / gt_sum:.2%}")
            print(f"  GT sample:    {gt_durations[:10]}")
            print(f"  Pred sample:  {[f'{x:.1f}' for x in pred_durations[:10]]}")
            print(f"  Log pred range: [{log_dur_pred.min():.3f}, {log_dur_pred.max():.3f}]")

    print(f"\n--- Average over {n} samples ---")
    print(f"  Avg GT sum:   {total_gt_sum / n:.0f}")
    print(f"  Avg pred sum: {total_pred_sum / n:.0f}")
    print(f"  Avg ratio:    {total_pred_sum / total_gt_sum:.2%}")


if __name__ == "__main__":
    main()
