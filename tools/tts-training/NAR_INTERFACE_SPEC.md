# NAR TTS 系统接口规范（冻结）

**版本**: 1.0  
**冻结日期**: 2026-07-26  
**状态**: NAR 产品主线基础规范

---

## 1. 文本输入

### 1.1 输入单位
- **基本单位**: 拼音（带声调）
- **序列化**: 声母 / 韵母 / 声调 分离为独立 token
- **声调**: 1-4 声 + 轻声（标记为 5）
- **示例**: "春眠不觉晓" → `ch un 1 m ian 2 b u 4 j ue 2 x iao 3`

### 1.2 特殊处理
- **多音字**: 使用 G2P 前端消歧，取上下文正确读音
- **儿化**: 不建模儿化，按字面拼音处理
- **变调**: 不建模变调（"一""不"按字典标注）
- **标点**: 不进入模型，转为结构标记

### 1.3 诗歌结构标记
```
<POEM_START>
春眠不觉晓 <LINE_END>
处处闻啼鸟 <LINE_END>
夜来风雨声 <LINE_END>
花落知多少 <POEM_END>
```

- `<LINE_END>`: 行末停顿 token，duration predictor 学习停顿长度
- `<POEM_START>` / `<POEM_END>`: 全诗边界
- 五言/七言节奏: 通过 duration 模板或外部 duration 控制，不硬编码到模型输入

---

## 2. 音频特征

### 2.1 采样率
- **教师音频**: 24000 Hz（CosyVoice 3 输出）
- **学生音频**: 24000 Hz（保持一致）
- **不重采样**

### 2.2 Mel 频谱
| 参数 | 值 |
|------|-----|
| 采样率 | 24000 Hz |
| FFT 大小 | 2048 |
| Hop length | 300 (≈12.5ms) |
| Win length | 1200 (50ms) |
| Mel bins | 80 |
| F_min | 0 Hz |
| F_max | 12000 Hz |
| 归一化 | 对数 Mel，mean-variance normalization |

### 2.3 F0（基频）
- 提取: pyworld (DIO + StoneMask)
- 鸣值: 非零 F0 为有声，零为无声
- 归一化: 对数域 mean-variance normalization（仅非零帧）

### 2.4 Energy
- 定义: 每帧 Mel 的 dB 值之和
- 归一化: mean-variance normalization

### 2.5 核心不变量
```
sum(duration_per_phoneme) == mel_length
mel_length == len(f0) == len(energy)
mel_frame_rate == sample_rate / hop_length == 80 frames/sec
```

---

## 3. Duration

### 3.1 单位
- 每个拼音音素的 Mel 帧数（hop_length=300, 80 frames/sec）

### 3.2 来源
- 训练: forced alignment（MFA 或 CTC segmentation）
- 推理: 模型预测 OR 外部指定

### 3.3 双模式支持
| 模式 | 用途 |
|------|------|
| 预测模式 | 模型 duration predictor 输出 |
| 外部模式 | 手动指定每音素 duration（产品级节奏控制）|

### 3.4 结构 token duration
- `<LINE_END>`: 由模型学习，典型值 20-40 帧（250-500ms）
- `<POEM_START>`: 固定 1 帧
- `<POEM_END>`: 固定 1 帧

---

## 4. 模型接口

### 4.1 输入
```
phoneme_ids: [B, L]  # 拼音音素序列 (int)
duration: [B, L]     # 外部模式时提供 (int, mel frames)
                     # 预测模式时为 None
```

### 4.2 输出
```
mel: [B, T, 80]      # Mel 频谱 (float)
f0: [B, T]           # F0 (float)
energy: [B, T]       # Energy (float)
duration: [B, L]     # 预测的 duration (仅预测模式)
```

### 4.3 Vocoder 接口
```
mel: [B, T, 80] → HiFi-GAN → waveform: [B, T * hop_length]
```

---

## 5. ONNX 导出规范

### 5.1 声学模型 ONNX
- 输入: `phoneme_ids` (动态序列长度)
- 可选输入: `duration` (外部模式)
- 输出: `mel`
- 动态轴: batch_size, sequence_length, mel_length

### 5.2 Vocoder ONNX
- 输入: `mel`
- 输出: `waveform`
- 动态轴: batch_size, mel_length

### 5.3 Android 部署
- 推理框架: ONNX Runtime (Android AAR)
- 声学模型 + vocoder 分别导出，端上串联
- 文本前端（G2P）在 Java/Kotlin 实现

---

## 6. 验收标准

### 6.1 硬性产品指标
| 指标 | 标准 |
|------|------|
| 整句缺失 | 0 次 |
| 整句重复 | 0 次 |
| 随机早停 | 0 次 |
| 同输入稳定性 | bitwise 一致（确定性生成）|
| 测试集完整率 | ≥ 95% |
| 推理 RTF (CPU) | < 1.0 (比实时快) |

### 6.2 质量指标
| 指标 | 目标 |
|------|------|
| CER | ≤ teacher baseline + 3% |
| Duration MAE | 待阶段 3 校准 |
| F0 correlation | ≥ 0.5 |
| 行末停顿存在率 | ≥ 90% |

---

## 7. 评估复用

使用已有评估管线 (`eval_tts.py`)，补充 NAR 专属指标:
- Duration MAE
- 总帧数误差
- 行末停顿误差
- F0 correlation
- RTF
- 峰值内存
- ONNX/PyTorch 差异

---

## 8. 未决问题（不阻塞 NAR）

- MOSS audio-head label/logit alignment（研究性问题，独立于 NAR）
- 儿化和变调建模（首版不实现）
- 多说话人/多风格（首版单风格）
