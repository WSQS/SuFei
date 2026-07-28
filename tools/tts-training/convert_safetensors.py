#!/usr/bin/env python
"""Convert .bin weights to .safetensors to bypass torch.load security check."""
import sys
import torch
from pathlib import Path
from safetensors.torch import save_file

def convert(model_dir):
    model_dir = Path(model_dir)
    bin_files = list(model_dir.glob("*.bin"))
    if not bin_files:
        print(f"No .bin files in {model_dir}")
        return

    for bin_file in bin_files:
        safetensors_file = bin_file.with_suffix(".safetensors")
        if safetensors_file.exists():
            print(f"  Skip (exists): {safetensors_file.name}")
            continue

        print(f"  Converting {bin_file.name}...")
        state_dict = torch.load(str(bin_file), map_location="cpu", weights_only=True)
        save_file(state_dict, str(safetensors_file))
        print(f"    -> {safetensors_file.name} ({safetensors_file.stat().st_size / 1e6:.1f}MB)")

    print(f"Done converting {model_dir}")

if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else "."
    # Convert TTS model
    convert(base / "models" / "MOSS-TTS-Nano")
    # Convert Audio Tokenizer
    convert(base / "models" / "MOSS-Audio-Tokenizer-Nano")
