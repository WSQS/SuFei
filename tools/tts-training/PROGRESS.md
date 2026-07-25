# TTS Training Data Pipeline

使用 CosyVoice 3 0.5B 生成高质量诗歌朗读音频，微调 MOSS-TTS-Nano 用于端侧部署。

## 架构

```
CosyVoice 3 (教师, WSL)
    ↓ 生成 WAV ✅
build_train_jsonl.py
    ↓ 构建 train_raw.jsonl ✅
MOSS-Audio-Tokenizer-Nano (Windows)
    ↓ WAV → audio_codes ✅
sft.py 微调 MOSS-TTS-Nano (Windows, RTX 4060)
    ↓ 3 epoch, ~30s ✅
verify.py
    ↓ 生成对比 WAV ✅
导出 ONNX → Android 部署
    ⏳ 待做
```

## 当前状态

| 步骤 | 状态 | 说明 |
|---|---|---|
| CosyVoice 3 数据生成 | ✅ | 51 首 WAV (19 古诗十九首 + 32 唐诗) |
| 训练数据构建 | ✅ | 51 条 (text + audio_codes) |
| MOSS-TTS-Nano 微调 | ✅ | 3 epoch, loss 4.7153→4.7069 |
| 验证对比 | ✅ | 3 首 base vs finetuned WAV |
| 导出 ONNX | ⏳ | |
| Android CI 集成 | ⏳ | PR #38 引擎已就绪 |

## 文件说明

### 生产脚本

| 文件 | 说明 |
|---|---|
| `generate_poetry.py` | 批量唐诗生成 (CosyVoice 3, WSL) |
| `generate_19poems.py` | 古诗十九首生成 (含拼音修正) |
| `run_generate.sh` | WSL 运行包装 |
| `run_19poems.sh` | WSL 运行包装 (十九首) |
| `setup_wsl.sh` | WSL 一键环境安装 |
| `build_train_jsonl.py` | WAV → MOSS 训练 JSONL |
| `convert_safetensors.py` | .bin → .safetensors (绕过 torch.load 安全检查) |

### 文档

| 文件 | 说明 |
|---|---|
| `SETUP_NOTES.md` | WSL + CUDA + uv 环境搭建踩坑指南 |
| `PROGRESS.md` | 本文件，进度跟踪 |

### 微调流程 (Windows, venv_moss)

```bash
# 1. 准备环境 (一次性)
uv venv venv_moss --python 3.12
uv pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
uv pip install transformers==4.57.1 peft accelerate soundfile sentencepiece safetensors numpy

# 2. 构建 JSONL (从 WAV)
python build_train_jsonl.py

# 3. 编码 audio codes
python MOSS-TTS-Nano/finetuning/prepare_data.py \
    --codec-path models/MOSS-Audio-Tokenizer-Nano \
    --input-jsonl data/train_raw.jsonl \
    --output-jsonl data/train_with_codes.jsonl \
    --skip-reference-audio-codes

# 4. 转换 .bin → .safetensors (绕过 torch 2.5 安全检查)
python convert_safetensors.py .

# 5. 训练
python -m accelerate.commands.launch \
    MOSS-TTS-Nano/finetuning/sft.py \
    --model-path models/MOSS-TTS-Nano \
    --codec-path models/MOSS-Audio-Tokenizer-Nano \
    --train-jsonl data/train_with_codes.jsonl \
    --output-dir output/moss_poetry_sft \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 1e-5 \
    --num-epochs 3 \
    --mixed-precision bf16 \
    --max-length 1024 \
    --channelwise-loss-weight 1,32

# 6. 验证
python MOSS-TTS-Nano/finetuning/verify.py \
    --checkpoint output/moss_poetry_sft/checkpoint-last \
    --mode continuation \
    --text "春眠不觉晓..." \
    --audio-tokenizer-pretrained-name-or-path models/MOSS-Audio-Tokenizer-Nano \
    --output-audio-path output/verify.wav
```

## 硬件

| 组件 | 规格 |
|---|---|
| GPU | RTX 4060 Laptop 8GB |
| RAM | 64GB |
| 训练峰值显存 | ~3.5GB (batch=1, max_length=1024, bf16) |
| 训练时间 | ~30s (51 首, 3 epoch) |

## 踩坑经验

### torch.load 安全检查 (CVE-2025-32434)
- transformers 4.57+ 要求 torch >= 2.6 才能用 `torch.load`
- MOSS 模型权重是 `.bin` 格式
- **解决**: 用 `convert_safetensors.py` 把 `.bin` 转 `.safetensors`，删除 `.bin`

### Windows 路径分隔符
- HuggingFace `from_pretrained` 把 Windows `\` 路径当 repo ID
- **解决**: `--audio-tokenizer-pretrained-name-or-path` 参数用 `/` 替换 `\`

### 网络代理
- Clash Verge 代理在 `127.0.0.1:7890`
- uv/pip 默认不走系统代理
- **解决**: `$env:HTTPS_PROXY = "http://127.0.0.1:7890"`
