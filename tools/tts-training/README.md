# TTS Training Pipeline

CosyVoice 3 → MOSS-TTS-Nano 微调 → 端侧诗歌 TTS。

## 状态：微调闭环已跑通 ✅

```
CosyVoice 3 (教师) → WAV → Audio Tokenizer → JSONL → SFT → 验证
                                                    ✅ 全链路打通
```

## 目录结构

```
tools/tts-training/
├── generate_poetry.py       # 唐诗批量生成 (WSL)
├── generate_19poems.py      # 古诗十九首生成 (含拼音修正)
├── build_train_jsonl.py     # WAV → MOSS 训练 JSONL
├── convert_safetensors.py   # .bin → .safetensors
├── setup_wsl.sh             # WSL 一键安装
├── run_generate.sh          # WSL 运行包装
├── run_19poems.sh           # WSL 十九首运行
├── SETUP_NOTES.md           # 环境搭建踩坑指南
├── PROGRESS.md              # 进度跟踪
└── .gitignore
```

## 详细文档

- [PROGRESS.md](PROGRESS.md) — 进度、参数、踩坑
- [SETUP_NOTES.md](SETUP_NOTES.md) — WSL 环境搭建完整指南

## 关键 PR

- PR #40: 本管线代码
- PR #38: MOSS-TTS-Nano Android 引擎 + CI
- PR #34: 系统 TTS 调参（基线）
