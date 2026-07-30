# TTS 训练数据生成环境搭建指南

本文档记录在 Windows 11 + RTX 4060 Laptop (8GB VRAM) + WSL2 上搭建
CosyVoice 3 训练数据生成环境的完整过程和踩坑经验。

## 硬件

| 组件 | 规格 |
|---|---|
| CPU | AMD Ryzen 9 7945HX (16C/32T) |
| GPU | NVIDIA RTX 4060 Laptop (8GB VRAM) |
| RAM | 64GB DDR5-5600 |
| OS | Windows 11 + WSL2 Ubuntu 20.04 |

## 为什么不用 Windows 原生

多次尝试在 Windows 上运行 TTS 模型（MOSS-TTS 8B、CosyVoice 3）均失败：

1. **`piper-phonemize` 无 Windows wheel** — CosyVoice 的 Matcha-TTS 依赖它
2. **`distutils` 在 Python 3.12 中被移除** — Matcha-TTS 的 setup.py 依赖它
3. **`bitsandbytes` 与 `transformers` 版本冲突** — `_is_hf_initialized` 参数不被旧版 bnb 支持
4. **HuggingFace 路径分隔符 bug** — Windows 的 `Path()` 把 `/` 变成 `\`，破坏 repo ID
5. **`onnxruntime` 缺少 `libcudnn.so`** — Windows 的 CUDA 库路径与 Linux 不同

**结论：中文 NLP/语音开发用 WSL2，不要在 Windows 原生环境折腾。**

## WSL2 网络修复

WSL2 的默认 apt 源 (`archive.ubuntu.com`) 在中国大陆经常超时。

```bash
# 切换阿里云镜像
sudo sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list
sudo sed -i 's|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list
```

如果 `apt-get update` 仍然超时，可能是 `unattended-upgrades` 进程占用锁：
```bash
sudo killall unattended-upgrade
sudo rm -f /var/lib/apt/lists/lock /var/cache/apt/archives/lock /var/lib/dpkg/lock-frontend
sudo dpkg --configure -a
```

## 用 uv 创建环境（推荐）

CosyVoice 要求 Python 3.10（不能用 3.12，因为 `distutils` 被移除）。
Windows 上的 uv 下载 Python 3.10 需要从 GitHub 下载，速度较慢。
在 WSL 中用 uv 更快。

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 创建 Python 3.10 虚拟环境
uv venv ~/cosyvoice-venv --python 3.10
source ~/cosyvoice-venv/bin/activate
```

## PyTorch 安装

### 版本选择

| torch 版本 | CUDA | 问题 |
|---|---|---|
| 2.3.1 | cu121 | 缺少 `torch.library.register_fake`，transformers 4.51.3 报错 |
| 2.4.1 | cu121 | 缺少 `libcudnn.so.9` |
| **2.5.1** | **cu124** | ✅ 可用，自带所有 CUDA 依赖 |

**关键经验：用 `uv pip install torch` 而非指定版本。** uv 会从 PyPI 安装
带 CUDA 的 wheel（约 2.5GB），内含所有 `nvidia-*` 依赖包。
不要用 `--no-deps`，否则缺少 cuDNN 库。

### 安装命令

```bash
# 不要用 --index-url（阿里云 PyTorch 镜像不是 PEP 503 simple index）
# 不要用 download.pytorch.org（中国太慢）
# 直接用阿里云 PyPI 镜像（有带 CUDA 的 torch wheel）
uv pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://mirrors.aliyun.com/pypi/simple/
```

### 版本冲突陷阱

CosyVoice 的 `requirements.txt` 指定了 `torch==2.3.1`。
如果先装 torch 2.5.1，再装 requirements，uv 会把 torch 降回 2.3.1。

**解决：先装 requirements，再强制升级 torch。**

```bash
# 1. 装 CosyVoice 依赖（不含 torch）
sed -i '/^torch==/d' requirements.txt
sed -i '/^torchaudio==/d' requirements.txt
uv pip install -r requirements.txt

# 2. 最后装 torch 2.5.1（覆盖被降级的版本）
uv pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://mirrors.aliyun.com/pypi/simple/
```

## CosyVoice 3 ��装

### clone + Matcha-TTS

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git ~/CosyVoice
cd ~/CosyVoice/third_party/Matcha-TTS
uv pip install -e .
```

### 依赖安装陷阱

1. **`conformer==0.3.2` 不在 PyPI 上** — 通过 Matcha-TTS 的 `pip install -e .` 安装
2. **`openai-whisper` 需要 `pkg_resources`** — 先装 `setuptools<70`（新版移除了 `pkg_resources`）
3. **`openai-whisper` 的构建隔离环境缺少 setuptools** — 用 `--no-build-isolation`
4. **`protobuf==4.25` 版本冲突** — 用 `--index-strategy unsafe-best-match` 让 uv 跨索引查找

```bash
# 完整安装顺序
uv pip install "setuptools<70" wheel
cd ~/CosyVoice
sed -i '/^torch==/d; /^torchaudio==/d; /^openai-whisper/d' requirements.txt
uv pip install -r requirements.txt \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --index-strategy unsafe-best-match
uv pip install openai-whisper==20231117 \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --no-build-isolation
uv pip install torch==2.5.1 torchaudio==2.5.1 \
    --index-url https://mirrors.aliyun.com/pypi/simple/
```

### ONNX Runtime CUDA 修复

CosyVoice 的 speech_tokenizer 使用 ONNX Runtime。
默认安装的 `onnxruntime-gpu==1.18.0` 需要 cuDNN 8，
但 torch 2.5.1 自带 cuDNN 9。

```bash
# 升级到支持 cuDNN 9 的版本
uv pip install onnxruntime-gpu==1.19.2

# 设置 LD_LIBRARY_PATH 让 ONNX Runtime 找到 cuDNN
CUDNN_LIB=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')")
export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"
```

## 模型下载

```bash
cd ~/CosyVoice
source ~/cosyvoice-venv/bin/activate
python -c "
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
"
```

ModelScope 在国内速度远优于 HuggingFace。

## CosyVoice 3 API 要点

### `<|endofprompt|>` 是必须的

CosyVoice 3 的所有推理方法都需要 `<|endofprompt|>` 标记：

```python
# inference_instruct2 的 instruct_text 格式
instruct_text = "You are a helpful assistant. 请用自然的方式朗读。<|endofprompt|>"

# inference_zero_shot 的 prompt_text 格式
prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
```

缺少 `<|endofprompt|>` 会报 `AssertionError`。

### 用 AutoModel 而非 CosyVoice3

```python
from cosyvoice.cli.cosyvoice import AutoModel
model = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

### 短文本限制

24 字的五言绝句可能因 flow decoder 的 CausalConv1d 最小长度限制而失败。
解决方案：在文本前加上标题和作者作为前缀。

## 性能数据

| 指标 | 数值 |
|---|---|
| 模型加载 | ~14 秒 |
| VRAM 占用 | ~4-5 GB |
| 每首诗生成 | ~10-30 秒（含推理） |
| 320 首预计 | ~2-3 小时 |
