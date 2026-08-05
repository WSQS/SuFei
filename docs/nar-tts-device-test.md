# NAR TTS 设备测试指南

## 准备模型文件

将以下 3 个文件复制到设备 `/sdcard/SuFei/models/nar/` 目录：

```
tools/tts-training/models/paddlespeech_onnx/
├── fastspeech2_csmsc_onnx_0.2.0/
│   ├── fastspeech2_csmsc.onnx     (142MB) → /sdcard/SuFei/models/nar/
│   └── phone_id_map.txt            (3KB)  → /sdcard/SuFei/models/nar/
└── hifigan_csmsc_onnx_0.2.0/
    └── hifigan_csmsc.onnx          (50MB)  → /sdcard/SuFei/models/nar/
```

### ADB 推送

```bash
adb shell mkdir -p /sdcard/SuFei/models/nar
adb push tools/tts-training/models/paddlespeech_onnx/fastspeech2_csmsc_onnx_0.2.0/fastspeech2_csmsc.onnx /sdcard/SuFei/models/nar/
adb push tools/tts-training/models/paddlespeech_onnx/fastspeech2_csmsc_onnx_0.2.0/phone_id_map.txt /sdcard/SuFei/models/nar/
adb push tools/tts-training/models/paddlespeech_onnx/hifigan_csmsc_onnx_0.2.0/hifigan_csmsc.onnx /sdcard/SuFei/models/nar/
```

## 安装 APK

```bash
./gradlew installDebug
```

## 测试

### 方式 1：Instrumented Test

```bash
./gradlew connectedAndroidTest --tests "*.NarOnnxEngineTest"
```

### 方式 2：手动测试

1. 打开 SuFei app
2. 进入任意诗歌详情页
3. 调用 `speakNar("春眠不觉晓，处处闻啼鸟。")`

### 预期结果

- FS2 推理：< 1 秒
- HiFiGAN 推理：1-2 秒
- 总延迟：< 3 秒
- 音频质量：CER ~7%（基于 PaddleSpeech 基线）
- 采样率：24kHz mono

## 已知限制

- 模型文件 192MB 不在 APK 内，需手动推送
- 首次推理需要加载 ONNX 模型（~2-3 秒）
- 推理在 Dispatchers.Default 上运行，不阻塞 UI
- 无标点停顿优化（PaddleSpeech 原生处理）
