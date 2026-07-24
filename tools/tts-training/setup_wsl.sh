#!/bin/bash
set -e

# ── CosyVoice 3 setup in WSL ──────────────────────────
# Run: wsl -d Ubuntu -- bash /mnt/c/.../tools/tts-training/setup_wsl.sh

export PATH="$HOME/.local/bin:$PATH"

# Install uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create Python 3.10 venv
echo "Creating venv..."
uv venv ~/cosyvoice-venv --python 3.10
source ~/cosyvoice-venv/bin/activate

# Install setuptools (needed by Matcha-TTS build)
uv pip install "setuptools<70" wheel

# Clone CosyVoice
if [ ! -d ~/CosyVoice ]; then
    echo "Cloning CosyVoice..."
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git ~/CosyVoice
fi

# Install Matcha-TTS (provides conformer)
echo "Installing Matcha-TTS..."
cd ~/CosyVoice/third_party/Matcha-TTS
uv pip install -e .

# Install CosyVoice deps (excluding torch, deepspeed, openai-whisper)
echo "Installing CosyVoice deps..."
cd ~/CosyVoice
sed -i '/^torch==/d; /^torchaudio==/d; /^deepspeed/d; /^openai-whisper/d' requirements.txt
uv pip install -r requirements.txt \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --index-strategy unsafe-best-match

# Install openai-whisper (needs --no-build-isolation for pkg_resources)
echo "Installing openai-whisper..."
uv pip install openai-whisper==20231117 \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --no-build-isolation

# Install PyTorch 2.5.1 last (prevents version downgrade)
echo "Installing PyTorch..."
uv pip install torch==2.5.1 torchaudio==2.5.1 \
    --index-url https://mirrors.aliyun.com/pypi/simple/

# Fix ONNX Runtime CUDA
echo "Fixing ONNX Runtime..."
uv pip install onnxruntime-gpu==1.19.2 \
    --index-url https://mirrors.aliyun.com/pypi/simple/

# Download model
echo "Downloading model..."
cd ~/CosyVoice
python -c "
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
print('Model download DONE')
"

# Verify
echo "Verifying..."
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
cd ~/CosyVoice
python -c "from cosyvoice.cli.cosyvoice import AutoModel; print('CosyVoice import: OK')"

echo "=== Setup complete ==="
