#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
source ~/cosyvoice-venv/bin/activate

CUDNN_LIB=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__)+'/lib')")
export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"

cd ~/CosyVoice
python /mnt/c/Users/wsqsy/Documents/android/SuFei/tools/tts-training/generate_19poems.py
