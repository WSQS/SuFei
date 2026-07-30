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

### Pitch/Energy Loss 分解���断

`diagnose_pitch_loss_decompose.py` 对 v2 checkpoint 做了帧级 loss 的四条件分解：

| 条件 | Pitch L1 | Energy L1 | 说明 |
|------|---------|----------|------|
| D: GT dur + predict mean=0 (baseline) | 0.659 | 0.796 | "什么都不预测" |
| **C: GT dur + GT phoneme mean (FLOOR)** | **0.303** | **0.506** | 理论下限 |
| B: GT dur + pred pitch (predictor only) | 0.596 | 0.755 | 隔离 predictor 误差 |
| A: pred dur + pred pitch (TRAINING) | 0.646 | 0.756 | 训练实际条件 |

**Floor 占比**：pitch 57%，energy 67%。帧级 loss 大部分是音素内 contour 变化造成的不可降噪声。

### GT-var vs E2E 对比实验：确认 train/inference gap 是核心问题

`eval_gtvar_train_cer.py` 在 D300fix GT-var checkpoint 上测了三���推理条件：

| 条件 | CER | 说明 |
|------|-----|------|
| **A: GT dur + GT pitch/energy** | **32.1%** | ≈ P2 baseline 31.8%，decoder 无容量问题 |
| B: GT dur + pred pitch/energy | 99.6% | predictor 输出对 decoder 是 OOD |
| C: pred all | 99.7% | 完全推理条件 |

**关键结论**：decoder 本身没有容量问题。32.1% → 99.7% 的 67pp gap 完全来自
predicted variance 的分布偏移。差异项是 pitch/energy predictor，不是 decoder 容量。

### Warm-start 实验（失败）

尝试从 D300fix GT-var checkpoint warm-start，freeze decoder 3k steps 训练 predictors，
然后解冻切换到 `--full_e2e`。

| 模型 | GT-all CER | Pred-all CER | 说明 |
|------|-----------|-------------|------|
| D300fix GT-var (baseline) | **32.1%** | 99.7% | decoder 完好，gap = 67pp |
| Warmstart v2 step24k | **32.1%** | 99.7% | decoder 保住了，但 gap 没缩小 |
| Full E2E v2 | — | **77.5%** | 最好的 pred-var 结果 |

`nar_train.py` 新增参数：`--warm_start`（重置 step 计数器 + predictors）、
`--reset_predictors`、`--freeze_steps`、`--full_e2e_steps`（分阶段切换模式）。

**结论**：7.6M decoder 无法同时适应 GT variance 和 pred variance 两种输入分布。
要么保住 GT 能力（32%），要么适应 pred（77.5%），没有中间态。

### Phoneme-level Loss 实验（v3）

将 pitch/energy loss 从帧级改为音素级，消除 57%/67% 不可降 floor。
`pool_to_phoneme()` 函数将帧级 GT 按 GT duration 池化为音素级均值，
再与 predictor 输出做 L1。

| 指标 | v2 (frame loss) | v3 (phoneme loss) |
|------|----------------|-------------------|
| pitch loss @ 24k | ~0.53 | **~0.14** (4x 下降) |
| energy loss @ 24k | ~0.76 | **~0.14** (5x 下降) |
| train mel L1 @ 24k | 0.345 | 0.358 |
| val mel L1 @ 24k | 0.911 | 0.920 |
| **pred-all CER** | **77.5%** | **84.2%** (更差 6.7pp) |

**Pitch/energy loss 大幅下降但 CER 反而恶化。**

### 根因分析：Decoder Shortcut 问题

代码级分析发现 v3 CER 恶化的根因是 **decoder shortcut**：

1. **梯度量级失衡**：phoneme-level loss (~50 音素) 的梯度比 frame-level (~500 帧) 小 10x，
   mel loss 完全主导 encoder 更新
2. **Decoder 忽略 pitch/energy embed**：mel loss 通过 `mel_input = x + pitch_embed + energy_embed`
   回传梯度时，decoder 发现最有效的降 mel loss 方式是直接从 phoneme embedding 推断 mel pattern
   （"shortcut"），而非利用 pitch/energy embed
3. **v2 的隐式正则**：frame-level loss 虽然有 floor，但其大梯度强迫 encoder 为 pitch/energy
   产生有区分度的表征，间接防止了 decoder shortcut
4. **v3 失去这个正则**：phoneme-level loss 梯度太弱，encoder 不再为 predictor 优化，
   decoder 走 shortcut → 频谱模糊 → CER 恶化

**代码位置**：
- 加法叠加：`nar_train.py:637` — `mel_input = mel_input + pitch_embed + energy_embed`
- mel loss 在 normalized 域：`nar_train.py:649` — `mel_loss = L1(mel_pred, mel_norm)`
- predictor 梯度被 detach 切断（duration 路径）：`nar_train.py:608`

### Scheme D: 交替训练 v4（失败）

`--alternating`：偶数步只训 predictors（无 mel loss），奇数步只训 decoder
（predictors 在 `no_grad` 下运行，用 pred variance）。

| 指标 | v2 baseline | v4 alternating |
|------|------------|----------------|
| pred-all CER | 77.5% | **95.3%** |
| val_mel_l1 @ 24k | 0.911 | 0.916 |

**失败原因**：梯度解耦切断组件间协作信号，decoder 有效步数减半（12k vs 24k），
encoder 梯度冲突。代码已 revert。

### Scheme E: 乘性门控 v5（失败）

将 `mel_input = x + pitch_embed + energy_embed` 改为
`mel_input = x * sigmoid(pitch_embed) * sigmoid(energy_embed)`，
结构上强制 decoder 依赖 variance embed（sigmoid gate 控制信息通量）。

| 指标 | v2 baseline | v5 mulgate |
|------|------------|------------|
| pred-all CER | 77.5% | **83.2%** |
| val_mel_l1 @ 24k | 0.911 | 0.911 |
| 100% CER | 2/30 | 3/30 |

**失败原因**：sigmoid 门控将 mel_input 压缩到 [0, x] 范围，丢失了 pitch/energy 的
加性信息（绝对基频/能量值）。gate 趋近 0.5 附近时信息衰减严重，decoder 收到的
信号被"稀释"。mel L1 持平但 CER 更差，说明频谱细节进一步模糊。

代码已 revert 回加法。

### Scheme A: FiLM v6（失败）

用 FiLM (Feature-wise Linear Modulation) 替换加性 variance embed。
pitch/energy embed 经 `film_gen`（2-layer MLP）生成 4 层 decoder 的 γ/β，
每层后应用 `dec = (1+γ)*dec + β`。Zero-init 保证初始 = 恒等映射。

decoder input **移除**加性 pitch/energy embed，结构强制 decoder 只能通过 FiLM
调制获取 variance 信息。

| 指标 | v2 baseline | v6 FiLM |
|------|------------|---------|
| pred-all CER | 77.5% | **86.5%** |
| 100% CER | 2/30 | 3/30 |
| <15% CER | 0/30 | 0/30 |
| val_mel_l1 @ 24k | 0.911 | 0.911 |
| model params | 7.6M | 8.2M (+0.6M film_gen) |

**失败原因**：FiLM 的 zero-init γ/β 从恒等映射起步，需要逐步学习调制模式。
但 mel loss 的梯度优先优化 decoder 层权重本身，film_gen 的梯度信号微弱，
导致 decoder 在 FiLM 调制成熟之前已经固化为 phoneme-only 推理路径。
加性 embed 虽然有 shortcut 风险，但它从第一步就注入 variance 信息，
decoder 被迫在早期建立对 variance 的依赖。

代码已 revert 回加性 embed。

### 结论：Decoder Shortcut 假设被推翻

四个方案（D 交替训练、E 乘性门控、A FiLM、+v3 phoneme-level loss）
全部失败，没有一个比 v2 additive embed baseline 好：

| 实验 | pred-all CER | vs v2 |
|------|-------------|-------|
| **v2 Full E2E (baseline)** | **77.5%** | — |
| v3 Phoneme-level loss | 84.2% | +6.7pp |
| v5 Multiplicative gate | 83.2% | +5.7pp |
| v6 FiLM | 86.5% | +9.0pp |
| v4 Alternating | 95.3% | +17.8pp |

**核心发现**：decoder shortcut **不是** pred-all CER=77.5% 的瓶颈。
GT-all=32.1% 证明 decoder 能用好 GT variance，gap 来自 variance predictor
误差本身（pitch corr=0.40, energy corr=0.26 仍然很低）。
任何削弱 variance embed 信息注入的改动都会恶化 CER——decoder 需要更多
variance 信息，而非更"强制"的依赖。

**下一步方向**：放弃 decoder shortcut 线索，转向：
1. **扩容**：d_model 256→384（匹配 PaddleSpeech），验证容量假设
2. **改善 variance predictor**：更多数据、更好的 predictor 架构
3. **归档自研**：锁定 PaddleSpeech ONNX (CER=6.8%) 为生产方案

### Phase 0: 数据管线审计（reviewer 发现三层标签 bug）

外部 review 发现之前的"既定事实"中隐藏了三个数据 bug：

1. **Mel 天花板虚构**（31.8% → 真实 20.2%）：旧管线从 vocoded audio 重新提取 mel
   （`generate_paddlespeech_distillation_data.py:632`），用 librosa 默认 power=2.0 功率谱，
   而 HiFiGAN 期望 power=1.0 幅度谱。改用 teacher FS2 直接输出的 mel 后，
   teacher_mel→HiFiGAN→ASR = **20.2%**（30 首同集对照，6.8% 是不同评测协议的结果）。

2. **F0 时间轴 bug**（2.1x 拉伸）：`nar_extract_features.py:71` 用 `f0[start//256:end//256]`
   对齐 pyworld F0 到 mel 帧，但 pyworld dio 默认 5ms frame_period → 120 samples/frame（非 256）。
   F0 标签只用到真实曲线的前 47%，pitch corr=0.40 就是"慢变趋势碰巧相关"的水平。

3. **Duration 标签来自 MFA**（有噪声）：v2 数据的 duration 仍是从 MFA TextGrid 比例缩放来的，
   分布严重偏态（15.1% dur≤2，max=152 帧），不是 teacher 的确定性输出。

### Phase 1-2: 数据修复 + 经典配方重训（v7-v9）

#### v7: teacher_mel + 修复 F0 hop + 经典 FS2 配方（MFA duration）

替换旧管线三处 bug 后用经典配方训练（GT dur teacher forcing + frame-level pitch/energy loss）。

评估发现 `model.forward()` 的 LengthRegulator 有 pad bug（将超出 total_durs 的帧映射到
phoneme 0 而非置零），导致 model.forward eval CER=99.4%。
用训练 forward path 评估后：

| 条件 | CER | 说明 |
|------|-----|------|
| 天花板 (teacher_mel→HiFiGAN→ASR) | 20.2% | 同集 30 首 |
| **B: GT dur + GT var** | **23.5%** | decoder 接近天花板 |
| **A: GT dur + pred var** | **23.9%** | pitch/energy predictor 几乎无损（F0 修复后） |
| C: pred-all | 96.9% | duration gap 崩溃 |

GT dur 条件下 CER=23.9%，距天花板仅 3.7pp。pitch/energy predictor 修复 F0 后完全正常工作。
唯一残余瓶颈是 duration predictor 的 train/inference gap。

#### v8: full_e2e 在干净数据上（MFA duration）

用 `--full_e2e` 训练消除 duration gap。pred-all CER=**56.5%**（0/30 完全崩溃）。
full_e2e 的帧级错位导致 mel 质量下降（val_mel_l1 0.864 vs v7 的 0.758），
但 pred-all 从 96.9% 改善到 56.5%。

#### v9: teacher duration 标签 + 经典配方（最终方案）

从 teacher FS2 ONNX 内部提取 duration predictor 输出（通过给 ONNX graph 添加 Round 节点
作为额外输出）。teacher duration 天然满足 `sum(dur)==mel_frames`，分布均匀（dur≤2: 0%，max=30）。

| | MFA 标签 | Teacher 内部 |
|---|---|---|
| median | 6 | **9** |
| dur≤2 | 15.1% | **0.0%** |
| max | 152 | **30** |

v9 经典配方训练结果：

| 指标 | v7 (MFA dur) | v9 (teacher dur) |
|------|-------------|-----------------|
| mel loss @24k | 0.22 | **0.12** |
| val_mel_l1 @24k | 0.758 | **0.350** |

三条件评估：

| 条件 | CER | 说明 |
|------|-----|------|
| 天花板 | 20.2% | teacher_mel→HiFiGAN→ASR |
| **A: GT dur + GT var** | **19.9%** | ≈ 天花板 |
| **B: GT dur + pred var** | **20.1%** | pitch/energy predictor 无损 |
| **C: pred-all (完全推理)** | **24.0%** | 仅差天花板 3.8pp |

**三层标签 bug 全部修复后，7.6M FS2 pred-all CER=24.0%，8/30 样本 CER<15%。**
从最初的 77.5%（旧脏数据 full_e2e）改善 53.5pp。
dur ratio=0.79 仍有轻微偏短（残余 4.1pp gap 的来源）。

### 实验全览（同条件对比）

| 实验 | 数据 | 训练方式 | pred-all CER | 关键变量 |
|------|------|---------|-------------|---------|
| v2 | re-extracted mel, F0 bug, MFA dur | full_e2e | 77.5% | 旧管线 baseline |
| v3 | 同上 | phoneme loss | 84.2% | − |
| v4 | 同上 | alternating | 95.3% | − |
| v5 | 同上 | mulgate | 83.2% | − |
| v6 | 同上 | FiLM | 86.5% | − |
| v7 | teacher_mel, F0 fixed, MFA dur | classic | 96.9% | dur gap 崩溃 |
| v8 | 同 v7 | full_e2e | 56.5% | 帧级错位 |
| **v9** | **teacher_mel, F0 fixed, teacher dur** | **classic** | **24.0%** | **最终方案** |

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
