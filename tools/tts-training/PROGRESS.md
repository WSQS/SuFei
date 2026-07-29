# TTS Training Pipeline

端侧中文诗歌 TTS 的训练实验与生产部署。

## 当前生产方案

**PaddleSpeech FS2 ONNX + HiFi-GAN ONNX**（ADR-0007）

- CER = 6.8%，确定性推理，无早停/重复
- FS2 (142MB) + HiFi-GAN (50MB) = 192MB，用户通过 ADB 推送
- Kotlin G2P 端侧实现，268 音素词汇表

## 研究实验：从零训练 FS2

### 目标

训练 7.6M 参数的 FastSpeech 2 用于端侧部署，替代 192MB 的 PaddleSpeech 方案。

### 数据管线

```
诗歌文本 -> G2P (pypinyin)
  -> PaddleSpeech FS2 ONNX -> teacher mel
  -> PaddleSpeech HiFiGAN ONNX -> teacher audio
  -> 特征提取 (mel/f0/energy)
  -> MFA 对齐 -> per-phoneme durations
  -> ASR 质量过滤 (FunASR Paraformer)
  -> manifest (.jsonl) + features (.npz)
```

### 实验结果（基于 garbage duration 标签，需重新评估）

| 实验 | 数据 | 训练方式 | Holdout delta | Train delta |
|------|------|---------|--------------|-------------|
| H1 | 171首 | 完整诗, GT variance | +70.8% | +1.7% |
| H2 | 171首 | 混合窗口(50/25/25) | +69.9% | — |
| D300 | 300首 | 完整诗, GT variance | +69.9% | — |

> **重要更正**：以上所有实验使用了垃圾 duration 标签（见 KNOWN_ISSUES #6）。
> 91% 的非首音素 duration = 2帧，首音素占 80%+ 总帧数。
> 模型实际学到的是 pitch/energy -> mel 映射，而非 phoneme -> mel。
> 泛化失败的归因需在修复数据后重新评估。

### 诊断分析（2026-07-29）

逐层 trace 诊断（`diagnose_layers.py`, `diagnose_durations.py`）发现三个根本问题：

1. **Duration 标签全垃圾**：MFA TextGrid 字符匹配失败 + TextGrid/mel 时轴 1.39x 不匹配，
   导致 `build_durations` 返回 91% 默认值2帧，diff 修正堆到 phoneme[0]
2. **Pitch predictor 坍缩**：输出常数 (std=0.0000)，因 duration 错位导致训练信号无效
3. **66.7% mel 帧是静音**：teacher 音频前后静音被全部分配给首音素

训练集表现分析（`diagnose_train_perf.py`）：
- GT-variance 下 frames 0-559 全部是同一 phoneme embedding（帧间 L2=0）
- decoder 的帧级区分信息完全来自 pitch embed (std=0.38) + energy embed (std=0.33)
- 用修正后的 durations 推理同一训练样本，mel L1 从 0.42 恶化到 0.67
  （模型从未学过真正的 phoneme -> mel 映射）

### E2E Predictor 实验

去掉 `--gt_variance`，duration/pitch/energy predictor 与 acoustic backbone 联合训练。

**评估结果**（holdout_20, 4 checkpoints）：

| Step | P2 (GT recon) | GT-var CER | Pred-var CER | Gap    | Dur L1 |
|------|--------------|------------|-------------|--------|--------|
| 6k   | 27.2%        | 99.6%      | 99.2%       | -0.4%  | 7.4    |
| 12k  | 27.2%        | 99.7%      | 99.0%       | -0.7%  | 7.3    |
| 18k  | 27.2%        | 99.9%      | 99.3%       | -0.6%  | 7.4    |
| 24k  | 27.2%        | 99.5%      | 99.0%       | -0.5%  | 7.3    |

Duration predictor 预测总帧数仅为 GT 的约 31%（如 595 -> 186 帧），音频 3x 加速。

### 关键 bug 修复

1. 逗号 G2P 映射 bug（已修复，KNOWN_ISSUES #1）
2. Duration 标签全垃圾（已修复，KNOWN_ISSUES #6）：
   - TextGrid/mel 时轴 1.39x 不匹配 + MFA 字符匹配失败
   - `build_durations_fixed()`: 比例缩放 TextGrid 时间戳到 mel_len
   - 91% 非首音素 dur=2 -> 3.2%，首音素占比 80% -> 1%
   - 所有 manifest 已重建（paddle_distill/train_171/train_300/holdout_20）

### 修复后重训评估

| 实验 | 训练方式 | Train pred-var CER | Holdout CER | Dur L1 | val mel L1 |
|------|---------|-------------------|-------------|--------|-----------|
| D300fix GT-var | gt_variance | — | 95.6% (GT-var) | — | 0.540 |
| D300fix E2E | e2e (GT dur 展开) | 98.7% | 99.6% | 5.7 | 0.810 |
| **Full E2E** | **full_e2e (pred dur 展开)** | **80.6%** | **98.7%** | **2.4(train)/5.4(holdout)** | **0.921** |

#### 训练/推理 gap 分析

诊断发现旧 E2E 训练存在严重的 train/inference gap：
- 训练时 length regulator 用 **GT durations** 展开，decoder 从未见过 predicted duration 序列
- 推理时用 predicted durations，帧级偏移累积导致 mel 对齐完全崩溃
- 三配置分解（训练样本 poem_0001）：
  - Config A (GT dur + GT var): mel L1 = 0.197
  - Config B (GT dur + Pred var): mel L1 = 0.192 (pitch/energy predictor 几乎完美)
  - Config C (Pred dur + Pred var): mel L1 = 0.732 (**duration 是唯一罪魁**)

#### Full E2E 训练（--full_e2e）

新增 `--full_e2e` 模式：训练时 decoder 用 predicted durations 展开，消除 gap。

训练集 pred-var CER 从 98.7% 降到 80.6%，100%-CER 样本从 18/30 降到 2/30。
但 CER 仍然太高（不可部署），holdout 仍然 ~99%。

**结论：ADR-0008 确认有效。** Duration bug 修复 + Full E2E 训练缩小了 train/inference
gap，但 7.6M FS2 从 300 首诗仍然无法泛化。训练集 pred-var CER=80.6% 也远高于
PaddleSpeech 的 ~7%。

#### 瓶颈分解（Full E2E 模型，训练集 30 首）

| 层级 | CER | 增量 | 说明 |
|------|-----|------|------|
| P2 (GT mel -> vocoder -> ASR) | 31.8% | 基线 | teacher 音频质量限制 |
| GT-var (model mel, perfect dur/pitch/energy) | 71.0% | +39pp | **acoustic model 容量不足** |
| Pred-var (model mel, predicted everything) | 80.6% | +9.6pp | duration predictor 残余误差 |

- mel L1: GT-var=0.311 vs Pred-var=0.319（差距仅 0.008，gap 已消除）
- 但 67% 的 mel 帧是静音 -> mel L1 被静音帧掩盖，实际语音帧质量远差于 L1 数字
- Duration ratio (pred/gt) = 0.76，仍系统性偏短 24%
- mel L1 vs CER 相关系数 = 0.365（脱钩确认）

**核心瓶颈是 acoustic model 容量不足**，不是 duration/pitch/energy predictor。
即使给模型完美的 duration + pitch + energy（GT-var），CER 仍达 71%。
PaddleSpeech FS2（37.3M, d_model=384）在同一管线下达 6.8% CER。

### 下一步方向

1. **预训练初始化**：从 PaddleSpeech FS2 权重 (37.3M, d_model=384) warm-start
2. **更好的 teacher**：用 CosyVoice 3 生成更高质量诗歌音频
3. **放弃泛化**：将目标诗全部放入训练集，只优化已知诗的表���

## 历史方案

### MOSS-TTS-Nano（已归档）

自回归 token 预测方案，因 token absorbing states 归档为 NO-GO。详见 ADR-0006。

### CosyVoice 3 + MOSS-TTS-Nano（早期实验）

51 首诗的 CosyVoice 3 teacher audio -> MOSS-Audio-Tokenizer -> MOSS-TTS-Nano fine-tune。
微调成功但生成质量不足。

## 环境

### 远程训练设备 (rtx)

| 组件 | 规格 |
|------|------|
| GPU | RTX 4090 24GB |
| CPU | i9-13900KF |
| Python | `C:\Users\wehao\anaconda3\envs\sufei-tts\python.exe` |
| 工作目录 | `E:\sufei-training\` |
| SSH | `wsl -e ssh rtx`（IP: 192.168.188.81）|
| 文件传输 | copyparty `http://192.168.188.81:3923/sufei/` |
| 后台任务 | `fdx bg run -- wsl -e ssh rtx "cmd.exe /c ..."` |

### 本地评估

| 组件 | 规格 |
|------|------|
| Python | `C:\Users\wsqsy\.conda\envs\open-webui\python.exe` |
| MFA | `C:\Users\wsqsy\.conda\envs\mfa\Scripts\mfa.exe` (v3.4.1) |
| CosyVoice | WSL Ubuntu `/home/sophomore/CosyVoice/` (Fun-CosyVoice3-0.5B) |

### Mel 参数（冻结）

```
sr=24000, n_fft=2048, hop=300, win=1200
n_mels=80, fmin=80, fmax=7600
```

## 相关文档

| 文件 | 说明 |
|------|------|
| [ADR-0006](../../docs/decisions/ADR-0006-nar-tts-architecture.md) | NAR TTS 架构选择 |
| [ADR-0007](../../docs/decisions/ADR-0007-on-device-nar-tts-paddlespeech.md) | PaddleSpeech 生产方案 |
| [ADR-0008](../../docs/decisions/ADR-0008-fs2-from-scratch-generalization-ceiling.md) | 从零训练 FS2 泛化失败 |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | 已知问题与修复记录 |
| [NAR_INTERFACE_SPEC.md](NAR_INTERFACE_SPEC.md) | NAR 接口规范 |
