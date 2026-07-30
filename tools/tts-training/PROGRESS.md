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
- Duration ratio (pred/gt) = 0.76，仍系统性偏短 24%
- mel L1 vs CER 相关系数 = 0.365（脱钩确认）

#### 静音帧 vs 语音帧 L1 分离分析（GT-var，27647 帧 / 30 首）

| 指标 | 静音帧 (67.6%) | 语音帧 (32.4%) |
|------|---------------|---------------|
| mel L1 (normalized) | 0.592 | 0.637 |
| 能量偏差 (pred - gt) | +1.5 dB（偏响） | -1.2 dB（偏轻） |
| 频谱对比度 (std across bins) | — | GT 的 **54-59%** |

按 GT 能量五分位分析：

| 分位 | 能量范围 | 帧数 | L1 | pred_e | gt_e |
|------|---------|------|-----|--------|------|
| Q1（静音） | [-11.5, -7.9] | 5530 | 0.647 | -7.94 | -9.45 |
| Q2 | [-7.9, -6.4] | 5529 | 0.564 | -6.65 | -7.08 |
| Q3 | [-6.4, -5.4] | 5529 | 0.571 | -5.96 | -5.88 |
| Q4 | [-5.4, -4.3] | 5529 | 0.590 | -5.35 | -4.86 |
| Q5（语音） | [-4.3, -0.9] | 5530 | 0.661 | -4.55 | -3.36 |

**关键发现**：原假设（静音帧 L1≈0、语音帧 L1≈0.9）被推翻。实际是：
1. **全面失败，非选择性偷懒**：静音帧 L1=0.592，语音帧 L1=0.637，差异极小
2. **向均值回归**：静音帧能量偏高 1.5 dB，语音帧偏低 1.2 dB。模型在压缩动态范围
3. **频谱模糊**：语音帧频谱对比度仅 GT 的 54-59%，共振峰结构丢失
4. **L1 误差质量**：65.9% 来自静音帧（因为数量多），34.1% 来自语音帧

**修正结论**：mel L1=0.311 并没有被静音帧"欺骗"——实际是之前用整体 mel L1=0.311 评估时
没有分离分析，而真正的 L1 分布是静音/语音帧都在 0.6 左右。整体 mel L1 之所以看起来低
(0.311)，是因为 mel normalization 的 std 较大（~1.0），实际 L1 为 0.6 对应的绝对误差
在可接受范围内，但频谱细节丢失使得 ASR 无法识别。

**核心瓶颈是 acoustic model 容量不足**，不是 duration/pitch/energy predictor。
即使给模型完美的 duration + pitch + energy（GT-var），CER 仍达 71%。
PaddleSpeech FS2（37.3M, d_model=384）在同一管线下达 6.8% CER。
三个症状都指向欠拟合：动态范围压缩、频谱模糊、L1 ��能量正相关。

### 下一步方向

1. **预训练初始化**：从 PaddleSpeech FS2 权重 (37.3M, d_model=384) warm-start
2. **更好的 teacher**：用 CosyVoice 3 生成更高质量诗歌音频
3. **放弃泛化**：将目标诗全部放入训练集，只优化已知诗的表���

### 远程 manifest 损坏发现 + 重训（v2）

#### 损坏发现

诊断 pitch/energy predictor 时发现：远程 rtx 上的 `train_300_manifest.jsonl` 是**旧损坏版**。
本地修复版 duration 分布健康（median=7, dur=2 仅 3.2%），但远程版本 89.9% 的音素 dur=2，
第一个音素吃掉大部分帧（如 poem_0001: `[560, 2, 2, 2, ...]`）。

**根因**：`build_durations_fixed` 中多音素字符匹配逻辑存在 bug——匹配到第一个 "宫" 后 tg_idx
推进，第二个音素搜索同字时跳到诗句中下一个出现的 "宫"，导致后续所有字符全部 NOT FOUND，
分配默认 0.05s ≈ 2 帧。这个 bug 在 `fix_durations.py` 本地调试时被修复并重新生成了 manifest，
但修复后的 manifest 没有同步到远程 rtx，远程训练一直用的是损坏版。

Pitch/energy predictor 在损坏 manifest 上的表现（correlation ≈ 0）证实了诊断：
帧级 GT 与音素预测完全错位，梯度是纯噪声。

#### Full E2E v2 实验（修复版 manifest, 24k steps）

同步正确 manifest 后重训 `--full_e2e`，其余参数不变。

| 指标 | v1 (坏 dur) | v2 (好 dur) | 变化 |
|------|-----------|-----------|------|
| Train pred-var CER | 80.6% | **77.5%** | -3.1pp |
| Holdout CER | 98.7% | 99.0% | ~持平 |
| Pitch correlation | 0.041 | **0.403** | **10x** |
| Energy correlation | 0.034 | **0.264** | **8x** |
| Pitch phoneme L1 | 0.759 | 0.614 | -19% |
| Energy phoneme L1 | 0.888 | 0.636 | -28% |
| val_mel_l1 @ 24k | 0.921 | 0.911 | ~持平 |

Pitch/energy predictor 大幅改善（correlation 10x/8x），但 CER 几乎没变。

**最终结论**：duration 数据损坏确实严重影响了 variance predictor 的学习，修复后 predictor
相关性大幅提升。但 CER 改善有限（train -3pp, holdout 持平），**acoustic model 容量不足
仍是核心瓶颈**。7.6M decoder 无法生成足够清晰的 mel，即使 variance predictor 学得更好，
decoder 也无法利用这些信息生成可识别的语音。

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
