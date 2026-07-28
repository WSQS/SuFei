"""Vocode overfit test mel with HiFi-GAN and compare to ground truth audio."""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
HIFIGAN_DIR = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
OVERFIT_DIR = ROOT / "output" / "overfit"
DURATION_JSON = ROOT / "data" / "duration_labels.json"


def main():
    poem_id = "poem_0285"

    mel_pred = np.load(str(OVERFIT_DIR / f"{poem_id}_mel_pred.npy"))
    mel_gt = np.load(str(OVERFIT_DIR / f"{poem_id}_mel_gt.npy"))

    print(f"mel_pred: {mel_pred.shape}, range=[{mel_pred.min():.2f}, {mel_pred.max():.2f}]")
    print(f"mel_gt:   {mel_gt.shape}, range=[{mel_gt.min():.2f}, {mel_gt.max():.2f}]")

    # Load HiFi-GAN
    sess = ort.InferenceSession(str(HIFIGAN_DIR / "hifigan_csmsc.onnx"))

    # Vocode predicted mel
    audio_pred = sess.run(None, {"logmel": mel_pred.astype(np.float32)})[0]
    audio_pred = audio_pred.flatten()

    # Vocode GT mel
    audio_gt = sess.run(None, {"logmel": mel_gt.astype(np.float32)})[0]
    audio_gt = audio_gt.flatten()

    print(f"\naudio_pred: {audio_pred.shape}, duration={len(audio_pred)/24000:.1f}s")
    print(f"audio_gt:   {audio_gt.shape}, duration={len(audio_gt)/24000:.1f}s")

    # Save
    sf.write(str(OVERFIT_DIR / f"{poem_id}_audio_pred.wav"), audio_pred, 24000)
    sf.write(str(OVERFIT_DIR / f"{poem_id}_audio_gt.wav"), audio_gt, 24000)

    # Also get original teacher audio
    with open(DURATION_JSON, encoding="utf-8") as f:
        dur_data = json.load(f)
    for s in dur_data["samples"]:
        if s["poem_id"] == poem_id:
            teacher_path = s["audio_path"]
            if Path(teacher_path).exists():
                teacher_audio, sr = sf.read(teacher_path)
                sf.write(str(OVERFIT_DIR / f"{poem_id}_audio_teacher.wav"), teacher_audio, sr)
                print(f"teacher:   {len(teacher_audio)} samples, sr={sr}, duration={len(teacher_audio)/sr:.1f}s")
            break

    print(f"\nSaved to {OVERFIT_DIR}")
    print("Files:")
    for f in sorted(OVERFIT_DIR.glob(f"{poem_id}*")):
        print(f"  {f.name} ({f.stat().st_size/1024:.0f}KB)")


if __name__ == "__main__":
    main()
