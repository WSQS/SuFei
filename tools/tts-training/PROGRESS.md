# TTS Training Data Pipeline

使用 CosyVoice 3 0.5B 生成高质量诗歌朗读音频，微调 MOSS-TTS-Nano 用于端侧部署。

## 背景

### 问题

系统 TTS 和通用小模型（Kokoro 82M、MOSS-TTS-Nano 100M）在诗歌朗读场景下质量不足：
- 节奏不自然（忽快忽慢）
- 缺乏情感和韵律感
- 多音字发音错误

### 方案：知识蒸馏（教师→学生）

用一个强大的教师模型（CosyVoice 3 0.5B）生成高质量朗读音频，
再用这些音频微调一个适合 Android 端侧部署的小模型（MOSS-TTS-Nano 100M）。

关键假设：诗歌朗读是一个极度收窄的领域（语料有限、风格一致），
专用微调后的 100M 模型可以达到远超通用 100M 模型的效果。

### 模型角色

| 角色 | 模型 | 参数量 | 运行环境 | 用途 |
|---|---|---|---|---|
| 教师 | CosyVoice 3 (Fun-CosyVoice3-0.5B-2512) | 0.5B | WSL + GPU | 生成高质量朗读音频 |
| 学生 | MOSS-TTS-Nano | 0.1B | Windows GPU 训练 / Android 推理 | 学习教师的朗读风格 |
| 编解码器 | MOSS-Audio-Tokenizer-Nano | 0.02B | Windows GPU | 把教师 WAV 转为学生训练用的 token 序列 |

## 核心概念

### train_raw.jsonl

训练管线的输入文件，每行一条 JSON 记录，描述一个训练样本：

```json
{
  "audio": "./audio/01_6105b29267b5.wav",
  "text": "客从远方来，遗我一端绮。相去万余里，故人心尚尔。",
  "language": "zh"
}
```

| 字段 | 说明 |
|---|---|
| `audio` | 教师生成的 WAV 文件路径 |
| `text` | 诗歌原文（去除注释、换行） |
| `language` | 语言代码 |
| `instruction` | 可选，风格指令（如"自然、克制的朗读方式"） |
| `ref_audio` | 可选，参考音频（用于音色克隆训练） |

此文件由 `build_train_jsonl.py` 从 CosyVoice 生成的 WAV 和 app 中的诗歌数据自动构建。

### audio_codes

MOSS-TTS-Nano 是一个**音频 token 自回归模型**——它不直接处理波形，
而是将声音编码为离散的 token 序列（类似语言模型中的文字 token）。

```
WAV 波形 → MOSS-Audio-Tokenizer-Nano → audio_codes (整数序列)
                                         ↓
text tokens + audio_codes → MOSS-TTS-Nano 训练 (下一个 token 预测)
```

audio_codes 的结构：
- 每帧（约 80ms 音频）对应 N 个 codebook 的整数
- N = 16（量化器数量，也叫 n_vq）
- 每个 codebook 的取值范围 ~1024

例如一段 5 秒的音频编码后可能有 ~62 帧 × 16 codebook = 992 个整数。

`prepare_data.py` 的工作就是用 Audio Tokenizer 把 `train_raw.jsonl` 中的每个 WAV 编码为
audio_codes，追加到每条记录中，输出 `train_with_codes.jsonl`：

```json
{
  "audio": "./audio/01_xxx.wav",
  "text": "客从远方来...",
  "language": "zh",
  "audio_codes": [[123, 456, ...], [789, 012, ...], ...]
}
```

### 为什么不直接用 WAV 训练？

| 维度 | 直接用 WAV | 用 audio_codes |
|---|---|---|
| 序列长度 | 5 秒音频 = 80,000 个采样点 | 5 秒音频 = ~62 帧 |
| 训练难度 | 极长序列，自回归不可行 | 序列长度可控（≤1024） |
| 模型设计 | 需要波形生成器 | 复用 LLM 架构（下一个 token 预测） |
| 推理效率 | 慢 | 快（生成 token 后一次性解码为 WAV） |

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
