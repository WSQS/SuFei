"""Convert all checkpoint .bin to .safetensors in output/ directory."""
import sys
import torch
from pathlib import Path
from safetensors.torch import save_file

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"

def convert_dir(model_dir: Path):
    bin_files = list(model_dir.glob("*.bin"))
    if not bin_files:
        return False
    for bin_file in bin_files:
        safetensors_file = bin_file.with_name("model.safetensors")
        if safetensors_file.exists():
            continue
        print(f"  Converting {bin_file.parent.parent.name}/{bin_file.parent.name}/{bin_file.name}...")
        state_dict = torch.load(str(bin_file), map_location="cpu", weights_only=True)
        save_file(state_dict, str(safetensors_file))
        print(f"    -> {safetensors_file.name} ({safetensors_file.stat().st_size / 1e6:.1f}MB)")
        bin_file.unlink()
    return True

if __name__ == "__main__":
    targets = [
        OUTPUT_ROOT / "moss_poetry_sft_wandb2" / "checkpoint-epoch-3",
        OUTPUT_ROOT / "moss_poetry_sft_320" / "checkpoint-epoch-1",
        OUTPUT_ROOT / "moss_poetry_sft_320" / "checkpoint-epoch-3",
        OUTPUT_ROOT / "moss_poetry_sft_320" / "checkpoint-epoch-5",
    ]
    for t in targets:
        if t.exists():
            convert_dir(t)
        else:
            print(f"  [SKIP] {t} not found")
    print("Done.")
