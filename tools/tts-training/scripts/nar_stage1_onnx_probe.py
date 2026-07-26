# nar_stage1_onnx_probe.py —
# Test FastSpeech2 + HiFiGAN ONNX inference without PaddlePaddle.
# Downloads pre-exported ONNX models and runs end-to-end on one poem.
# Uses onnxruntime + Python only.

import os
import sys
import time
import json
from pathlib import Path

# We'll need: onnxruntime, numpy, soundfile
# G2P via pypinyin (already in venv_moss potentially)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "paddlespeech_onnx"
OUTPUT_DIR = ROOT / "output" / "nar_probe"

# Pre-exported ONNX download URLs from PaddleSpeech
FS2_ONNX_URL = "https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_csmsc_onnx_0.2.0.zip"
HIFIGAN_ONNX_URL = "https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_onnx_0.2.0.zip"


def main():
    import onnxruntime as ort
    import numpy as np
    import soundfile as sf

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download and extract ONNX models
    fs2_zip = MODEL_DIR / "fastspeech2_csmsc_onnx_0.2.0.zip"
    hifigan_zip = MODEL_DIR / "hifigan_csmsc_onnx_0.2.0.zip"

    if not (MODEL_DIR / "fastspeech2_csmsc_onnx_0.2.0").exists():
        print("Downloading FastSpeech2 ONNX...")
        if not fs2_zip.exists():
            import urllib.request
            urllib.request.urlretrieve(FS2_ONNX_URL, fs2_zip)
        import zipfile
        with zipfile.ZipFile(fs2_zip, 'r') as z:
            z.extractall(MODEL_DIR)
        print("  Done")

    if not (MODEL_DIR / "hifigan_csmsc_onnx_0.2.0").exists():
        print("Downloading HiFiGAN ONNX...")
        if not hifigan_zip.exists():
            import urllib.request
            urllib.request.urlretrieve(HIFIGAN_ONNX_URL, hifigan_zip)
        import zipfile
        with zipfile.ZipFile(hifigan_zip, 'r') as z:
            z.extractall(MODEL_DIR)
        print("  Done")

    # Step 2: List model files
    fs2_dir = MODEL_DIR / "fastspeech2_csmsc_onnx_0.2.0"
    hifigan_dir = MODEL_DIR / "hifigan_csmsc_onnx_0.2.0"

    print("\nFastSpeech2 files:")
    for f in sorted(fs2_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f}MB)")

    print("\nHiFiGAN files:")
    for f in sorted(hifigan_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f}MB)")

    # Step 3: Load phone_id_map.txt
    phone_id_map = fs2_dir / "phone_id_map.txt"
    if phone_id_map.exists():
        print(f"\nPhone ID map sample:")
        with open(phone_id_map, encoding="utf-8") as f:
            lines = f.readlines()
            for l in lines[:10]:
                print(f"  {l.strip()}")
            print(f"  ... ({len(lines)} total)")

    # Step 4: Load ONNX models and check I/O
    print("\nLoading FastSpeech2 ONNX...")
    fs2_sess = ort.InferenceSession(str(fs2_dir / "fastspeech2_csmsc.onnx"))
    print("  Inputs:")
    for inp in fs2_sess.get_inputs():
        print(f"    {inp.name}: shape={inp.shape}, type={inp.type}")
    print("  Outputs:")
    for out in fs2_sess.get_outputs():
        print(f"    {out.name}: shape={out.shape}, type={out.type}")

    print("\nLoading HiFiGAN ONNX...")
    # Find the actual ONNX file
    hifigan_onnx = list(hifigan_dir.glob("*.onnx"))
    if hifigan_onnx:
        hifigan_sess = ort.InferenceSession(str(hifigan_onnx[0]))
        print("  Inputs:")
        for inp in hifigan_sess.get_inputs():
            print(f"    {inp.name}: shape={inp.shape}, type={inp.type}")
        print("  Outputs:")
        for out in hifigan_sess.get_outputs():
            print(f"    {out.name}: shape={out.shape}, type={out.type}")
    else:
        print("  No .onnx file found!")

    print("\n=== Stage 1 ONNX probe complete ===")
    print("Next: implement G2P + run end-to-end inference on a test poem")


if __name__ == "__main__":
    main()
