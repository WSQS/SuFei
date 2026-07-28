"""FastSpeech 2 inference: text → mel → audio.

Loads a trained checkpoint and generates speech from text input.
Supports both predicted-duration and external-duration modes.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nar_fastspeech2 import FastSpeech2

CHECKPOINT_DIR = ROOT / "checkpoints"
HIFIGAN_DIR = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
OUTPUT_DIR = ROOT / "output" / "nar_inference"

PHONE_ID_MAP = ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0" / "phone_id_map.txt"


def load_phone_id_map():
    phone_map = {}
    with open(PHONE_ID_MAP, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                phone_map[parts[0]] = int(parts[1])
    return phone_map


# Import G2P from manifest builder
from nar_build_manifest import text_to_phonemes


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = FastSpeech2(
        vocab_size=268,
        d_model=256,
        nhead=2,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        n_mels=80,
        dropout=0.0,  # eval mode
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = ckpt["stats"]
    return model, stats, ckpt.get("step", 0)


def inference(model, phone_ids, stats, device, duration_scale=1.0):
    """Run inference: phone_ids → mel."""
    mel_mean = torch.tensor(stats["mel_mean"], device=device)
    mel_std = torch.tensor(stats["mel_std"], device=device)

    phone_ids = phone_ids.unsqueeze(0).to(device)

    with torch.no_grad():
        mel_pred, log_dur_pred, pitch_pred, energy_pred = model(phone_ids)

        # Scale durations if needed
        if duration_scale != 1.0:
            # Re-run with scaled durations
            scaled_durations = (log_dur_pred.detach().exp() * duration_scale).round().clamp(min=1).long()
            mel_pred, _, _, _ = model(phone_ids, durations=scaled_durations)

    # Denormalize mel
    mel_denorm = mel_pred[0].cpu().numpy() * mel_std.cpu().numpy() + mel_mean.cpu().numpy()
    return mel_denorm


def vocode(mel, device="cpu"):
    """Mel → audio via HiFi-GAN ONNX."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(HIFIGAN_DIR / "hifigan_csmsc.onnx"))
    audio = sess.run(None, {"logmel": mel.astype(np.float32)})[0]
    return audio.flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "fs2_final.pt"))
    parser.add_argument("--text", default="春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。")
    parser.add_argument("--output", default=None)
    parser.add_argument("--duration_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    model, stats, step = load_checkpoint(args.checkpoint, args.device)
    print(f"  Trained steps: {step}")

    # G2P
    phone_map = load_phone_id_map()
    phonemes = text_to_phonemes(args.text)
    phone_ids_list = []
    for pstr, _ in phonemes:
        if pstr in phone_map:
            phone_ids_list.append(phone_map[pstr])
        else:
            print(f"  WARN: phone '{pstr}' not in vocab")

    phone_ids = torch.tensor(phone_ids_list, dtype=torch.long)
    print(f"  Text: {args.text}")
    print(f"  Phonemes: {len(phonemes)}, IDs: {len(phone_ids_list)}")

    # Inference
    t0 = time.time()
    mel = inference(model, phone_ids, stats, args.device, args.duration_scale)
    infer_time = time.time() - t0
    audio_dur = mel.shape[0] * 300 / 24000
    rtf = infer_time / audio_dur if audio_dur > 0 else float("inf")
    print(f"  Mel: {mel.shape}, range=[{mel.min():.2f}, {mel.max():.2f}]")
    print(f"  Inference: {infer_time:.3f}s, audio: {audio_dur:.1f}s, RTF: {rtf:.3f}")

    # Vocode
    t0 = time.time()
    audio = vocode(mel, args.device)
    voc_time = time.time() - t0
    print(f"  Vocoder: {voc_time:.3f}s, audio: {len(audio)} samples ({len(audio)/24000:.1f}s)")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.output or f"nar_inf_{int(time.time())}.wav"
    out_path = OUTPUT_DIR / out_name
    sf.write(str(out_path), audio, 24000)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
