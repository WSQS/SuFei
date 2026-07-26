"""Export FastSpeech 2 to ONNX for Android deployment.

Two modes:
1. Acoustic model: phone_ids → mel (with duration predictor)
2. Full pipeline: phone_ids → mel (using GT or predicted durations)

The exported model accepts:
  Input: phoneme_ids [L] (dynamic)
  Output: mel [T, 80] (dynamic)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nar_fastspeech2 import FastSpeech2

CHECKPOINT_DIR = ROOT / "checkpoints"
ONNX_DIR = ROOT / "models" / "fs2_onnx"


class FS2ForExport(torch.nn.Module):
    """Wrapper that runs full forward pass with predicted durations."""

    def __init__(self, model, mel_mean, mel_std):
        super().__init__()
        self.model = model
        self.register_buffer("mel_mean", mel_mean)
        self.register_buffer("mel_std", mel_std)

    def forward(self, phoneme_ids):
        # Run model with predicted durations
        mel_pred, log_dur_pred, pitch_pred, energy_pred = self.model(phoneme_ids.unsqueeze(0))

        # Denormalize mel
        mel_denorm = mel_pred[0] * self.mel_std + self.mel_mean
        return mel_denorm, log_dur_pred[0].exp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "fs2_final.pt"))
    parser.add_argument("--output", default=str(ONNX_DIR / "fastspeech2.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    device = "cpu"  # ONNX export works best on CPU

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FastSpeech2(
        vocab_size=268,
        d_model=256,
        nhead=2,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        n_mels=80,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = ckpt["stats"]
    mel_mean = torch.tensor(stats["mel_mean"], device=device)
    mel_std = torch.tensor(stats["mel_std"], device=device)

    # Wrap for export
    export_model = FS2ForExport(model, mel_mean, mel_std).to(device)
    export_model.eval()

    # Dummy input
    dummy_phonemes = torch.tensor([151, 87, 155, 15, 174, 183, 263, 182], dtype=torch.long)

    # Verify PyTorch output first
    with torch.no_grad():
        mel_out, dur_out = export_model(dummy_phonemes)
    print(f"PyTorch output:")
    print(f"  mel: {mel_out.shape}, range=[{mel_out.min():.2f}, {mel_out.max():.2f}]")
    print(f"  dur: {dur_out.shape}, sum={dur_out.sum():.0f}")

    # Export
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model,
        (dummy_phonemes,),
        args.output,
        input_names=["phoneme_ids"],
        output_names=["mel", "durations"],
        dynamic_axes={
            "phoneme_ids": {0: "L"},
            "mel": {0: "T"},
            "durations": {0: "L"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    print(f"\nONNX exported: {args.output}")
    print(f"  Size: {Path(args.output).stat().st_size / 1024 / 1024:.1f} MB")

    # Verify ONNX
    import onnxruntime as ort
    sess = ort.InferenceSession(args.output)
    mel_onnx, dur_onnx = sess.run(None, {"phoneme_ids": dummy_phonemes.numpy()})

    print(f"\nONNX output:")
    print(f"  mel: {mel_onnx.shape}, range=[{mel_onnx.min():.2f}, {mel_onnx.max():.2f}]")
    print(f"  dur: {dur_onnx.shape}, sum={dur_onnx.sum():.0f}")

    # Compare
    mel_diff = np.abs(mel_out.numpy() - mel_onnx).max()
    dur_diff = np.abs(dur_out.numpy() - dur_onnx).max()
    print(f"\n  Max mel diff: {mel_diff:.6f}")
    print(f"  Max dur diff: {dur_diff:.6f}")

    if mel_diff < 0.01:
        print("  ONNX export VERIFIED")
    else:
        print("  WARNING: ONNX output differs from PyTorch!")


if __name__ == "__main__":
    main()
