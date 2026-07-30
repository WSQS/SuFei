#!/bin/bash
# Run poetry audio generation in WSL
# Usage: wsl -d Ubuntu -- bash /mnt/c/.../tools/tts-training/run_generate.sh

set -e
export PATH="$HOME/.local/bin:$PATH"
source ~/cosyvoice-venv/bin/activate

# Set LD_LIBRARY_PATH for cuDNN (ONNX Runtime needs it)
CUDNN_LIB=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__)+'/lib')")
export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"

cd ~/CosyVoice
SCRIPT=/mnt/c/Users/wsqsy/Documents/android/SuFei/tools/tts-training/generate_poetry.py

# Parse args
MAX_POEMS="${MAX_POEMS:-320}"
MIN_LEN="${MIN_LEN:-20}"
INSTRUCTION="${INSTRUCTION:-natural}"

python "$SCRIPT" --max "$MAX_POEMS" --min-len "$MIN_LEN" --instruction "$INSTRUCTION"
