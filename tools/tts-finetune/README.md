# TTS Fine-tune Pipeline

Specialize MOSS-TTS-Nano (100M) for classical Chinese poetry recitation.

## Architecture

```
唐诗文本 (320首)
    ↓
MOSS-TTS 8B + 情感指令 → 高质量 WAV (训练数据)
    ↓
MOSS-Audio-Tokenizer → audio token sequences
    ↓
(text tokens, audio tokens) pairs
    ↓
LoRA fine-tune MOSS-TTS-Nano
    ↓
Export merged ONNX → deploy to Android
```

## Hardware

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 9 7945HX (16C/32T) |
| GPU | RTX 4060 Laptop 8GB VRAM |
| RAM | 64GB DDR5-5600 |

## Steps

1. `01_generate_data.py` — Batch generate WAV with MOSS-TTS 8B
2. `02_encode_tokens.py` — Encode WAV → audio tokens with Audio Tokenizer
3. `03_prepare_dataset.py` — Build (text_tokens, audio_tokens) training pairs
4. `04_finetune_lora.py` — LoRA fine-tune Nano model
5. `05_export_onnx.py` — Merge LoRA + export to ONNX

## Notes

- All scripts run on the local machine, not on Android
- 8B inference uses ~8GB VRAM (INT4 quantization via llama.cpp or bitsandbytes)
- LoRA fine-tune fits in 8GB VRAM with small batch + gradient checkpointing
- Output ONNX replaces the base model in `app/src/main/assets/`
