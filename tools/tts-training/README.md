# TTS Training Data Pipeline

使用 CosyVoice 3 0.5B 生成高质量诗歌朗读音频，用于微调端侧 TTS 模型。

## 架构

```
唐诗三百首 (320首)
    ↓
CosyVoice 3 0.5B + 指令控制 (WSL, RTX 4060)
    ↓
高质量 WAV (48kHz, ~2-3小时)
    ↓
Audio Tokenizer → token 序列
    ↓
LoRA 微调 MOSS-TTS-Nano → 端侧 ONNX
```

## 环境搭建

详见 [SETUP_NOTES.md](SETUP_NOTES.md)。

一键安装脚本（在 WSL 中运行）：
```bash
bash /mnt/c/Users/wsqsy/Documents/android/SuFei/tools/tts-training/setup_wsl.sh
```

## 生成数据

```bash
# 在 WSL 中运行
bash /mnt/c/Users/wsqsy/Documents/android/SuFei/tools/tts-training/run_generate.sh

# 或手动指定参数
source ~/cosyvoice-venv/bin/activate
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__)+\"/lib\")'):$LD_LIBRARY_PATH"
cd ~/CosyVoice
python /mnt/c/.../tools/tts-training/generate_poetry.py --max 320
```

## 文件说明

| 文件 | 说明 |
|---|---|
| `generate_poetry.py` | 批量生成脚本（加载唐诗 → CosyVoice 推理 → 保存 WAV） |
| `run_generate.sh` | WSL 运行包装脚本（设置环境变量 + 调用 Python） |
| `setup_wsl.sh` | WSL 环境安装脚本 |
| `SETUP_NOTES.md` | 环境搭建踩坑指南 |
