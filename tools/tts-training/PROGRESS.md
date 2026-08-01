# TTS Training Pipeline

端侧中文诗歌 TTS 的训练实验与生产部署。

## 当前生产方案

**SuFei FS2 ONNX (7.6M) + HiFi-GAN ONNX**（ADR-0007 + 自研蒸馏）

- pred-all CER = 24.0%（train），21.5%（holdout），teacher 天花板 20.2%
- ONNX E2E CER = 24.0%（与 PyTorch 零差距）
- FS2 (35MB) + HiFi-GAN (50MB) = **85MB 总计**（vs PaddleSpeech 192MB）
- 已部署验证：Lenovo TB-Q706F (Android 13)，G2P→FS2→HiFiGAN→AudioTrack 全链路跑通
- 模型路径：`getExternalFilesDir/models/nar/`（app 私有目录，无需存储权限）
- Kotlin G2P 端���实现，268 音素词汇表

**历史方案**：PaddleSpeech FS2 (142MB, CER=6.8%) 作为 fallback 保留在代码中。

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

### MAS Duration Learning（ADR-0009）：解耦 duration 外部依赖

**动机**：v9 的 duration 标签来自 teacher (PaddleSpeech FS2 ONNX) 内部导出，
数据源被锁死在 teacher 上。MAS (Monotonic Alignment Search) 把对齐从
"预处理步骤"变成"训练副产品"，只用 (音素序列, mel) 就能自动学出单调对齐。

架构：FS2 encoder 输出 + mel target 各经一个投影层 (256→128) 映射到共享空间，
计算 cost matrix → MAS DP 找最优单调对齐 → 硬整数 duration。
推理时投影层+MAS 全部丢弃，模型结构与 v9 完全一致。

#### M1：MAS 对齐质量验证（frozen v9 encoder）

`eval_mas_alignment.py`：用 v9 frozen encoder + 可训练投影层，验证 MAS dur
与 teacher dur 的相关性。

| 指标 | Random projections | Trained (2k steps) |
|------|-------------------|--------------------|
| Train per-sample corr | -0.070 | **0.589** |
| Holdout per-sample corr | -0.031 | **0.590** |
| MAS dur ratio (MAS/GT) | 1.000 | 1.000 |
| MAS dur≤2 | 49.4% | 4.7% (teacher: 0%) |

**判定：MARGINAL**（corr=0.59 < 0.7 阈值）。但 frozen encoder 从未为 alignment
优化过，0.59 是"碰巧能对齐"的水平。M2 联合训练后 corr 应提升。

#### v10a：MAS dur 展开 mel（失败 — duration 坍缩）

`nar_train_align.py` v1：Phase 1 (0-2k) teacher dur warmup，Phase 2 (2k+) MAS dur
用于 mel 展开 + dur predictor target。

| 指标 | v9 | v10a step8k |
|------|-----|------------|
| Train CER | 24.0% | **95.5%** |
| Holdout CER | 21.5% | **98.4%** |
| dur ratio (pred/teacher) | 0.79 | **0.201** |

**失败根因**：MAS dur 同时用于 mel 展开和 dur predictor target，形成正反馈。
推理时 dur predictor 输出坍缩到 teacher 的 20%，音频极度加速，ASR 全部失败。
训练日志中 dur_loss 看着很低（~0.01），但那是因为 MAS dur 和 pred dur
坍缩到了同一个短分布。

#### v10b：pred dur 展开 mel（失败 — 损失帧错位）

修复尝试：Phase 2 改用 **pred_dur** 展开 mel，MAS dur 只做 dur_loss target。

**失败根因**（step 9850 被终止）：pred-dur 展开使 mel/pitch/energy 三个损失
全部帧错位（full_e2e 的 H4 问题重现）。mel 卡在 0.25 平台（v9 是 0.15），
pitch/energy 从 0.33/0.42 涨到 0.60/0.65——错位噪声地板，不会恢复。
治标不治本：v10a 的病根是对齐坍缩，不是展开源。

**附带发现**：重启的运行 0 字节日志秒崩 = conda 环境 libiomp5md.dll 重复
（OMP Error #15），`set KMP_DUPLICATE_LIB_OK=TRUE` 规避。

#### v10c：先验 + forward-sum（部分有效 — 暴露双稳态）

按 RAD-TTS/FastPitch 配方加入 beta-binomial 对角先验（仅 DP 搜索）+
CTC forward-sum 损失，Phase 2 改回 **MAS-dur 展开**（总和恰=T，损失无错位）。

三次运行三种命运（fsum 平台 2.6 / 7.5 / 13），对齐优化是双稳态、
初始化运气主导：

| 运行 | 配置 | fsum@1k | pct_le2 | 结局 |
|------|------|---------|---------|------|
| 冒烟 1200 步 | 联合训练 | 2.6 | 0.05 | 运气好，收敛 |
| j240 全量 | 联合训练 | 7.5 | 0.72 | 坍缩（fsum 大梯度冲乱 encoder，mel 0.12→0.44） |
| j241 全量 | x.detach() + seed | 12.6 | 0.88 | 更差：两个裸线性投影在冻结特征上容量不足 |

**诊断**：联合训练时 encoder 被 fsum 重塑提供了 aligner 缺的容量（代价是
encoder 受损 + 看运气）；detach 保护了 encoder（mel 稳定 0.17）但裸线性
投影学不动。与 NeMo AlignmentEncoder 对照发现两处关键偏差 → v10d。

#### v10d：NeMo 式 aligner + 先验入 softmax + 相位闸门（训练中）

1. **Aligner 换卷积栈 + L2 距离得分**（temperature 5e-4，展开式省显存）——
   容量放进 aligner 自身，encoder 保持 detach 保护；
2. **先验加进 softmax 内部**——forward-sum/path-NLL/DP 三者共享同一份
   先验塑形分布（v10c 只给 DP，CTC 对原始偏斜得分自锁）；
3. **相位闸门**：`ema_pct_le2 < 0.20` 才切 MAS 展开，aligner 不健康时
   decoder 永远停留在安全的 teacher-dur warmup 模式。

结果（j242, seed 42, 24k 步, 121.7 min）：训练全程稳定——fsum 1.2→0.76，
闸门 step 2052 打开，MAS 相位 mel 瞬态 0.57 后回落至 0.128（v9=0.125），
pct_le2 收敛到 0.013，dur_corr 随训练自发升至 ~0.45。
**首个健康完赛的 MAS 版本。**

#### v10d 评估 + duration scale 补偿（2×2 对照）

| 模型 | Train CER | Holdout CER |
|------|-----------|-------------|
| teacher 天花板 | ~20.2% | — |
| v9 原始 | 24.0% | 21.5% |
| **v9 + dur×1.27** | **20.3%** | **19.4%** ← 全局最优 |
| v10d 原始 | 29.3% | 30.2% |
| v10d + dur×1.25 | 20.7% | 24.6% |

**发现 1**：dur ratio 0.79/0.80 的系统性偏短是 v9/v10 共同的主要 CER
贡献者——×1.27 补偿实验证明了这一点（v9: 24.0→20.3 / 21.5→19.4，双双
打到天花板）。根因有二：① log 域 L1 的最优解是 log 中位数（≈几何均值），
对右偏的时长分布系统性低估总和（高方差的停顿位置被拉向几何均值）；
② 训练目标 clamp(max=100) 截断——数据 66.7% 帧是静音，长停顿动辄
200-300 帧，predictor 学到的世界里停顿最长 100 帧。
**决定（2026-07-31）**：×1.27 是治标（均匀缩放拉长普通音节、极长停顿
仍不足），不落地生产；按根因修复立项——显式停顿建模（韵律边界 token +
停顿单独预测，与 M3 韵律工作合并），辅以移除截断/换损失域的对照实验。

**发现 2（M2 裁决）**：公平对比（双方补偿后）train 平价（20.7 vs 20.3），
holdout 差 5.2pp——MAS 标签独立性的当前溢价，纯粹来自对齐质量的泛化。
对症实验 v10f：align_warmup 4000（aligner 成熟再开闸）+ 30k 步。

#### v10e：MAS 训练提速 16.4x

v9=24.3 step/s vs v10=3.2 step/s（慢 7.6x）根因：`maximum_path_np` 纯
Python 逐格双循环（~64 万次迭代/步）+ 每步 8 次 GPU 同步。修复：DP 改为
按帧扫描、音素维 numpy 向量化（51.3ms→3.1ms，16.4x，20 组随机矩阵路径
逐位等价）+ log_prob 批量一次 .cpu()。无新依赖，训练期 only。

### 下一步方向（更新 2026-07-31，M2 达成后）

1. **Duration 偏短根因修复**（不采用 ×1.27 落袋）：显式停顿/韵律边界
   token 建模（停顿不再靠 predictor 从音素上下文猜），对照实验：移除
   clamp(max=100)、损失域调整。与 M3 韵律工作合并推进
2. **M3 数据源解锁**：用 v10f 配方接入 CosyVoice / 真人朗读音频
   （生成 → ASR 门控 + best-of-N 选优 → MAS 自动对齐训练），
   无需任何外部对齐工具——韵律超越 teacher 的正式起点
3. **工程遗留**：checkpoint 文件名加 run 名（防覆盖）；fsum 的 8 次
   CTC 调用批量化（10.7 → 预计 15+ step/s）

### M3 Phase 0：CosyVoice3 数据源冒烟（2026-07-31）

10 首诗（5 holdout + 5 train）× 2 风格，WSL Ubuntu 本机 GPU 生成
（Fun-CosyVoice3-0.5B，统一 prompt 约定 `指令<|endofprompt|>转写`），
rtx 评估。脚本：`m3_gen_cosyvoice.py` / `m3_phase0_assess.py` /
`m3_mel_calibration.py`。

**Mel 提取惯例校准（一次性、永久有效）**：teacher 原生 mel → HiFiGAN →
音频 → 五种候选惯例提取回比。胜者 **power=1 + log10 + eps 1e-10**
（raw L1=0.102，仿射 a=0.982/b≈0 恒等）；旧 extract_mel（power=2+ln）
L1=4.44 差 43 倍，其拟合系数 0.228≈1/(2·ln10) 完全解释了历史 31.8%
天花板。此惯例自此为一切外部音源的标准提取。

**评估结果（20/20 生成成功，0 条 100% CER）**：

| 变体 | direct CER | ceiling CER（校准提取→HiFiGAN） |
|------|-----------|-------------------------------|
| plain | 18% | 17% |
| recite | 20% | 20% |

**发现 1（Phase 1 绿灯）**：ceiling ≈ direct（±3pp 内，常常相等）——
校准惯例下，跨说话人（CosyVoice 声音 ≠ 标贝）过 HiFiGAN CSMSC 的
声码损失≈0。**M3 训练可直接复用现有目标空间与 vocoder，无需微调。**

**发现 2**：direct 18-20% ≈ teacher 音频在同协议下的水平（~20%），
逐首方差大（3%-34%）→ best-of-N 选优有确定价值。

**发现 3（待听感裁决）**：recite 指令没有如预期放慢语速——反而普遍
更短（如 12.96s→9.40s），CER 均值略差。指令措辞需迭代，风格效果
以人耳判断为准（wav 留存 `data/m3_cosyvoice_phase0/`）。

### M3 Phase 1：CosyVoice 数据全管线 + 冷启动训练（2026-07-31）

管线全链路一次跑通：320 首（300 train + 20 holdout）× 2 候选 WSL 本机
生成（640/640 成功）→ rtx best-of-N ASR 选优 + CER≤45% 门控（320/320
全过）→ 校准惯例特征提取（mel/f0/energy，f0 用 pyworld
frame_period=12.5ms 原生帧率，无重采样）→ **冷启动** MAS 训练
（`--cold_start`：闸门关闭期只训 align+fsum+dur，闸门 open 后全量——
teacher duration 零依赖，49.5 min 完赛）。checkpoint: `m3_v1_final.pt`。

| 模型 | Train CER | Holdout CER | dur ratio |
|------|-----------|-------------|-----------|
| CosyVoice 直读 floor | ~18-20% | — | — |
| m3_v1 原始 | 32.7% | 34.3% | **0.789** |
| m3_v1 + dur×1.27（诊断） | **19.4%** | 28.7% | — |
| 参照：v9 原始 | 24.0% | 21.5% | 0.79 |
| 参照：v10f + ×1.25 | 20.7% | 22.6% | 0.80 |

**发现 1（管线验证 ✅）**：×1.27 补偿后 Train 19.4% 直接落到 CosyVoice
直读 floor（18-20%）——新声源 + 校准提取 + 冷启动 MAS + 零外部标签的
全链路成立，训练/提取侧没有额外损失。

**发现 2（偏短跨源三连）**：dur ratio 0.789 与 teacher-dur 线 (0.79)、
MAS 标贝线 (0.80) 完全一致——偏短与数据源/对齐方式无关，纯粹是训练目标
问题（log-L1 几何均值偏差 + clamp(max=100) 截断，见上文根因分析）。
M3 线生产化被 #7 根因修复阻塞：raw 32.7/34.3 不可部署，×1.27 不落地。

**发现 3（泛化差距）**：补偿后 holdout 28.7% vs train 19.4%
（gap 9.3pp；v10f 仅 1.9pp）——CosyVoice 单次采样的音色/韵律方差
远大于确定性 teacher + 对齐泛化不足。数据扩容（生成近乎免费）是
对症杠杆。

**待办**：听感裁决——M3 立项动机是韵律，CER 只是底线。wav 在 rtx
`output/m3_v1_eval/`（原始）与 `output/m3_v1_eval_s127/`（×1.27），
已拷回本地 `data/`。

### #7 Duration 偏短根因修复（2026-08-01，迭代中）

修复形态收敛为「**去 clamp + 时长总和一致性损失**」而非新增停顿 token——
标点在 G2P 里本就有专用音素（"，""。""？""！"各占一个 phone id），MAS
会把静音段分给它们，缺的只是让 predictor 能学到长时长的训练目标。

`nar_train_align.py` 新增：`--dur_clamp_max`（≤0 去上界截断）、
`--w_dursum`（`|Σexp(log_dur)−T|/T`，对 log_dur 的梯度权重是
exp(log_dur)，自动集中于最长 token 即停顿）、`--run_name`
（checkpoint 防覆盖，还掉工程债）。实现走 opencode ns-glm/glm-5.1，
spec 见 `M3_DURFIX_SPEC.md`。

| 运行 | 配置 | eval ratio | 零补偿 CER (T/H) | 补偿后 CER |
|------|------|-----------|-----------------|-----------|
| m3_v1 | clamp100，无 dursum | 0.789 | 32.7 / 34.3 | ×1.27 → 19.4 / 28.7 |
| m3_v2 | 去 clamp + w=0.1 | 0.803 | 32.9 / 35.6 | ×1.25 → 19.3 / 27.0 |
| m3_v3 | w=0.5 | 0.843 | 31.2 / 32.6 | — |
| **m3_v4** | **+ dropout-free 损失** | **0.994** | **21.6 / 30.3** | 无需补偿 |
| **m3_v5** | **+ w_dur_linear 0.03 + 1003 首** | **1.003** | **19.9 / 22.6** | 无需补偿 |

**裁决（2026-08-01）**：#7 根因修复达成——ratio 0.79→0.994，零补偿
CER 32.7/34.3→21.6/30.3，×1.27 权宜正式退场。训练末端 pred_ratio
0.99-1.01、dursum≈0.01，且 dur/mel 损失与前版持平（sum 一致性没有
牺牲逐音素拟合）。

**遗留观察**：m3_v4 零补偿(21.6/30.3) 与均匀缩放诊断上限
（m3_v2×1.25: 19.3/27.0）尚差 ~2-3pp——总时长已对，但**质量分布**
可能过度集中在停顿（exp 加权把缺口全给了最长 token，普通音节仍偏快）。
候选精修：sum 损失的 per-token 梯度上限调低（如 clamp 5.0≈148 帧），
迫使质量向中等音节摊薄。优先级低于数据扩容（holdout gap 8.7pp 才是
主要矛盾）。

**教训 1（拔河权重）**：w=0.1 恰好在停顿 token 上与 log-L1 梯度打平
（平衡点 w≈0.08），只推动 +0.014；w=0.5 在停顿上 6 倍胜出、普通音节
仍被 L1 锚住——选择性正确，ratio 推到 0.84。

**教训 2（Jensen 通胀，两次上当）**：train-mode dropout 使
E[exp(x+ε)] > exp(E[x])，噪声和比确定性和虚高 ~10%。第一次：训练日志
pred_ratio 显示 0.91 而 eval 实测 0.803（指标改为 eval-mode 重算修复）；
第二次：**dursum 损失本身也作用在噪声和上**——把虚高的和优化到 T，
确定性和就永远停在短 ~10% 的均衡点（m3_v3 停滞 0.84 的根因）。修复：
损失改为对 duration predictor 做 eval-mode 确定性第二前向（带梯度，
L1 保留 train-mode 当正则）→ m3_v4。

### 王字复读诊断：dursum 质量集中的可听化实证（2026-08-01）

用户报告 m3_v4 合成的 poem_0027 中「唐代·王维」的王字复读。诊断链：

1. **跨版本 ASR A/B**（`diag_asr_0027_v2.py`）：c0 源音频、m3_v1 raw、
   m3_v1×1.27 全部单王，仅 m3_v4 复读 → 回归隔离到 v4 的时长目标改动；
2. **逐音素时长导出**（`diag_dur_0027.py`，v1 vs v4）：全诗总和
   845→1067（目标 1063，sum 修复 ✓），但「·」映射的 "。" 音素
   5.8→**14.6 帧**、uang2（王）29.8→**37.6 帧**（0.47s）、uei2（维）
   21.9→30.3；
3. **源音频事实**：c0 连读「唐代王维」，· 处无停顿；训练时 MAS 把该
   "。" 挤在浊音区 → decoder 学到的停顿音素声学混入浊音成分。

**机制**：dursum 梯度 ∝ exp(log_dur) 把补回的时长质量堆到停顿与长元音；
推理时在源音频连读处插入 ~0.18s 停顿 + 0.47s 超长王 → 停顿段渲染出
王色彩浊音 + 重起音，听感即「王王维」。log-L1 无力抵抗：
log(37.6/29.8)=0.23 与短音素 4→5 帧同价。#7 的「遗留观察」由此从
低优先级升级为需修复项。

**修复（已实现，待 m3_v5 验证）**：`M3_DURFIX_SPEC.md` Addendum 3，
新增 `--w_dur_linear`——对确定性前向的 exp(log_dur) 帧域 smooth L1
（beta=2.0）锚到 MAS 目标，长 token 超 8 帧的代价是短 token 超 1 帧的
8 倍，迫使 sum 修正按比例摊薄。默认 0 位级不变；梯度平衡估算
w=0.03 在 30 帧 token 上与 dursum(0.5) 推力同阶。opencode 实现
（j289），已审查 + py_compile + 同步 rtx。

### 王字复读根因二段：「·」映射句号 token 的数据 bug（2026-08-01，m3_v6）

m3_v5 验证结果：**线性锚达成了它的目标但复读仍在**——uang2 回落到
28.8 帧（v1 水平），可 ASR 依旧「王王维」，且「·」对应的 "。" 反而
17.3 帧。说明这不是时长分布问题，而是**转写与音频不符**：

- 训练文本统一为「标题，朝代·作者。正文」，G2P 把 "·" 归一化成 "。"
  音素（`PUNCT_TO_PHONE`）；
- CosyVoice 朗读「唐代·王维」是**连读无停顿**的，MAS 只能把这个 "。"
  挤到浊音帧上——**全语料 1003 首每首 1-2 处**，句号音素的声学被系统性
  掺入浊音；
- 推理时 predictor 按语料统计（真句尾长停顿为主）给这个 "。" ~17 帧
  「停顿」，其被污染的声学在王字前渲染出浊音重起音 → 复读。
- v1 没复读只是因为 clamp+log-L1 把所有停顿都预测过短（5.8 帧），
  bug 被另一个 bug 掩盖了。v9 老管线不受影响：teacher 音频是照 "。"
  真停顿朗读的，token 与音频一致。

**修复（m3_v6）**：忠实转写——"·" 不产生任何音素 token。
`text_to_phonemes` 加 `punct_map` 参数（默认不变，v9 线不动）；
`m3_fix_dot_token.py` 对三份 manifest 做位置删除（G2P 重生成 1323/1323
与现存 ids 完全一致，验证可复现后按 "·" 字符位删 token，durations
按比例重算）：m3p2b_train（1003，删 1 token×925 + 2×78）、
m3p2b_holdout（20）、m3b_train（300，评估可比性用）。特征/归一化
统计为纯 wav 派生，全部复用。m3_v6 = m3_v5 配方仅换 manifest。
注意：部署时 app 侧 G2P 需同步「·→无 token」约定（或保留 ·→。，
届时会渲染成干净停顿而非浊音伪影，属可接受风格差异）。

**m3_v6 结果（2026-08-01）**：

| 模型（零补偿） | Train CER | Holdout CER | Gap |
|------|-----------|-------------|-----|
| m3_v5 | 19.9% | 22.6% | 2.7pp |
| **m3_v6（去 · token）** | **20.6%** | **21.0%** | **0.4pp** |
| 参照：v9 原始（生产） | 24.0% | 21.5% | −2.5pp |

- **复读根除**：0027 序列中 代/王 之间无 token，原 "。" 的 ~17 帧被
  MAS 归还给 代（d 3.7→15.3、ai4 15.6→23.1），王/维 28.4/31.8 不变，
  ASR 单王 ✓。真句尾停顿质量回位（新。：v5 错堆 in1=37.1/。=8.6，
  v6 为 in1=7.6/。=33.5）。
- **Holdout 21.0% 首超生产 v9（21.5）**，train 领先 4.1→3.4pp，泛化
  差距归零（0.4pp）——假停顿 token 的声学污染是系统性伤害，删除后
  holdout 再降 1.6pp。已进入 CosyVoice 直读 floor（18-20%）邻域。
- **M3 Phase 2 达成**：零外部时长标签 + 零推理补偿 + 全自动数据管线，
  两项 CER 指标均优于生产 v9。checkpoint: `m3_v6_final.pt`；
  评估音频已拷回本地 `data/m3_v6_eval/`（50 wav），待听感裁决
  （M3 立项动机是韵律，CER 只是底线）。后续里程碑：ONNX 导出 +
  fp16 + APK 内嵌（部署方案已评审，见 deployment 计划）。

### M3 Phase 2：数据扩容 320→1050 首 + m3_v5（2026-08-01）

- 选诗 `m3_select_phase2.py`：114,395 首过滤池 → 750 首新诗
  （唐宋、20-120 字、字符集/句末约束、对 phase1 去重 20,870 首），
  38,928 字 ≈ 227 分钟目标音频，G2P 零丢弃。
- **双机生成**（各 375 首 × 2 候选）：rtx 原生 conda 环境 1h32m
  完成（RTF≈0.43），本地 WSL 2h56m——4090 快 2 倍。两侧均
  n_fail=0，共 1500 wav ≈ 5.9h 音频。
- rtx CosyVoice 环境全程 opencode + 3 轮调试落库 `rtx_setup/`
  （Windows pip 三坑：openai-whisper 需 setuptools<81 +
  no-build-isolation；piper-phonemize 无 Windows 发行版；cmd2 3.5.1
  的 rich>=15 冲突 → 整组 Optuna 遗留剔除），可复现。
- prep（j291，12 min）：1070 首候选 → **1023 过门控**（max_cer=0.45，
  47 首全是 p2 新诗，phase-2 淘汰率 6.3%；phase-1 320 首全保留）。
  Train 1003 + Holdout 20（沿用固定 holdout_20_manifest_v3，与全部
  历史评估可比）。特征 .npz × 1023，norm stats 仅 train split。
  首跑 j290 因 rtx 上 `m3_prep_features.py` 陈旧（不识别
  `--extra_manifest`）exit 2，同步脚本后重跑即过。
- **m3_v5**（j292，54.6 min，30k 步 / 250 epochs @9.2 step/s）：
  m3_v4 配方 + 新数据 + `--w_dur_linear 0.03`（warm start v10f）。
  durlin 未加权值 2-3.8（前 2k 步）→ 0.4-1.0（尾部），×0.03 后与
  dursum 项同量级，权重标定一次通过；pred_ratio 全程 0.98-1.03。

| 模型（零补偿） | Train CER | Holdout CER | Gap | dur ratio |
|------|-----------|-------------|-----|-----------|
| m3_v4（300 首） | 21.6% | 30.3% | 8.7pp | 0.994 |
| **m3_v5（1003 首）** | **19.9%** | **22.6%** | **2.7pp** | **1.003** |
| 参照：v9 原始（生产） | 24.0% | 21.5% | −2.5pp | 0.79 |
| 参照：v9+scale 诊断 | 20.3% | 19.4% | −0.9pp | — |

**裁决（2026-08-01）**：数据扩容 3.2× + 线性锚定把 holdout 从 30.3
拉到 **22.6**（-7.7pp），泛化差距 8.7→2.7pp——Phase 1 的「方差大、
数据不够」判断成立。m3_v5 零补偿、零外部时长标签，已与生产 v9 的
holdout 持平（22.6 vs 21.5），train 反超 4.1pp。holdout 0/20 全错、
16/20 低于 30%。epoch 数只有 m3_v4 的 ~1/3（250 vs ~800），如需再压
可加步数。checkpoint: `m3_v5_final.pt`。

**0027 复检（m3_v5）**：线性锚生效——uang2 37.6→28.8 帧（回到 v1 的
29.8 水平），但 ASR 仍读出「唐代王王维」，复读**未消失**。根因升级为
数据层问题，见下节 m3_v6。

### v9 Holdout 泛化评估

| 集合 | CER | 说明 |
|------|-----|------|
| Train (30首) | 24.0% | 与之前评估一致 |
| **Holdout (20首)** | **21.5%** | 未见过的诗 |
| Gap | **-2.5pp** | 无过拟合 |

模型完全没有过拟合——holdout CER 甚至比 train 还低（小样本统计波动）。
300 首训练数据对泛化足够，24% CER 是真实泛化能力而非记忆。

### ONNX 导出精度验证

| 条件 | CER | 说明 |
|------|-----|------|
| PyTorch (30首) | 24.0% | 训练 forward path |
| **ONNX E2E (30首)** | **24.0%** | ONNX FS2 + HiFiGAN |
| Gap | **0.0pp** | 无精度损失 |

OnnxAttention（手动 Q/K/V + matmul）权重拷贝正确，ONNX Runtime 算子融合无副作用。
之前测的 32% CER 是因 `train[:10]` 只取了 10 首，样本量不足。
ONNX 模型 production-ready：35MB FS2 + 50MB HiFiGAN = 85MB。

#### m3_v6 ONNX 导出（2026-08-01，部署 task #12 第 1 步）

复用 v9 的 `Fs2OnnxWrapper`（架构相同，只是训练损失权重与 dot-fix manifest 不同），
脚本 `scripts/export_m3v6_onnx.py`，在**同一 holdout-20 / train-30（seed 42）协议**上
做 PyTorch↔ONNX 全链路对齐（FS2 ONNX → 反归一化 → HiFiGAN → ASR → CER）：

| 条件 | Train (30) | Holdout (20) |
|------|-----------|-------------|
| PyTorch (m3_v6 eval) | 20.6% | 21.0% |
| **ONNX E2E** | **20.6%** | **21.0%** |
| Gap | **−0.0pp** | **−0.0pp** |

零精度损失，与 v9 导出一致。产物在 `models/sufei_fs2_onnx_m3v6/`：
`fastspeech2_sufei_m3v6.onnx`(35.6MB, fp32) + `norm_stats.npz`(1.1KB) + `phone_id_map.txt`(2.1KB)。
实测 fp32 包体 = FS2 35.6MB + HiFiGAN 52.0MB = **87.6MB**（即之前记的「85MB」）。
下一步：fp16 量化（→~44MB）+ 精度复核。

#### m3_v6 fp16 量化（2026-08-01，部署 task #12 第 2 步）

脚本 `scripts/fp16_m3v6_onnx.py`（`onnxconverter_common.float16`，`keep_io_types=True`
保持 I/O fp32，glue 代码零改动）。**坑**：length regulator 的 `repeat_interleave`
导出成 `SplitToSequence`/`ConcatFromSequence`（序列类型张量），转换器不会把 fp16 传播进
序列元素类型 → 加载报 `seq(float) != seq(float16)`。修复：`op_block_list` 把序列算子族
钉在 fp32（它们无权重，不占体积；转换器自动在边界插 Cast），并在 `to_fp16` 内加 load 自检。

同一 holdout-20 / train-30 协议复评（fp32 基线 train 20.6 / holdout 21.0）：

| 配置 | Train | Holdout | 包体 |
|------|-------|---------|------|
| fp32 基线 | 20.6% | 21.0% | 87.6 MB |
| A: FS2-16 + HiFi-32 | 20.5% (−0.1) | 22.1% (+1.1) | 69.9 MB |
| **B: 双 fp16** | **20.3% (−0.3)** | **21.9% (+0.9)** | **50.3 MB** |

FS2 fp16 mel MAE=0.0004（几乎无损）；holdout +1pp 属 20 首噪声 + fp16 对 `exp(log_dur)`
取整使 3/10 首总帧长 ±1 帧（12.5ms，不可闻）。HiFiGAN fp16 无害（B 的 holdout 反而优于 A）。
**选定 config B（双 fp16，50.3MB，较 fp32 −43%）**。

**部署打包**（`scripts/package_m3v6_deploy.py` → `models/sufei_m3v6_deploy/`）：
config-B 文件重命名为 app 期望名 + sha256 完整性清单：
`fastspeech2_sufei.onnx`(17.9MB) + `hifigan_csmsc.onnx`(32.4MB) + `phone_id_map.txt` +
`norm_stats.npz` + `SHA256SUMS.txt` + `manifest.json`（含版本/量化/CER/G2P 约定）。
已 fetch 到本地 `tools/tts-training/models/sufei_m3v6_deploy/`（.onnx 已 gitignore）。
fp16 试听样本（A/B/fp32 对比）在 `data/m3_v6_fp16_eval/`。

#### app 侧集成（2026-08-01，已改，待编译/端侧验证）

发现 **文本格式失配**（比 · 约定更关键）：`DetailScreen.kt` 原 narText =
`title+dynasty+author` 空拼接 + content（去换行）——**缺** title 后的「，」和 author 后的「。」，
与 m3_v6 训练格式 `{title}，{dynasty}{author}。{content_flat}` 不符（dynasty+author
glued 是对的，正好对齐 ·-drop）。已改两处：
1. `DetailScreen.kt` narText → `"${title}，${dynasty}${author}。" + content.replace("\n","").replace(" ","")`；
2. `ChineseG2p.kt` PUNCT_MAP 删掉 `"·"→"。"`（· 不产 token）。
`<eos>` 已核对一致（训练 G2P line 169 `("<eos>", None)` = Kotlin `phones.add("<eos>")`）。

#### 部署方向决策（2026-08-01，用户拍板；实现留待后续）

**交付形态**：最终给用户是**内嵌**（离线即用），但内嵌在 **CI/CD 构建时**完成，
git 仓库**不追踪**这 50MB 模型（避免二进制进 git 历史的永久负担）。
**制品托管**：**HuggingFace**（pinned revision + sha256，公开可复现）。

**落地方案（已设计，未实现）**：
1. Gradle `fetchTtsModels` 任务：构建时从 HF 拉 pinned 制品 → 校验 sha256 →
   解到 `app/src/main/assets/models/nar/`（已存在且 sha256 匹配则跳过），挂在
   `merge*Assets` 前。**CI 与本地 fresh clone 行为一致**（clone 后首次构建自动补齐）。
2. `app/src/main/assets/models/nar/` 加入 `.gitignore`。
3. 加载：首启从 assets 拷到 `filesDir/models/nar/`，复用现有**已在联想真机验证过**的
   `createSession(path)`，**零引擎改动**（代价 ~50MB 设备存储；比改 `byte[]` 加载低风险）。
4. release APK +~50MB（用户接受）。

**状态：推进暂停，留待后续。**
- 已完成：ONNX 导出(0.0pp) / fp16 config B(50.3MB) / 打包+sha256 /
  app 文本格式失配修复 + G2P ·-同步（已编译通过，**改动在工作树中，未 commit**）。
- 待做：① HF 仓库 + 上传 `models/sufei_m3v6_deploy/`（pinned + sha256）；
  ② `fetchTtsModels` gradle 任务；③ 首启 assets→filesDir 拷贝；
  ④ 端侧 E2E 复验（Kotlin↔Python G2P 逐音素 parity 仅 v9 验过，本次动了 ·+文本格式）。

### 端侧部署验证（Lenovo TB-Q706F, Android 13）

| 步骤 | 结果 |
|------|------|
| 模型推送 | `getExternalFilesDir/models/nar/`（app 私有目录） |
| 初始崩溃 | `/sdcard/` 路径 EACCES (errno 13) → 改用 app 私有目录 |
| G2P | 176 音素 ID（`suFei=true`） |
| FS2 推理 | mel 1406×80 |
| HiFiGAN | 421,800 samples ≈ 17.6s |
| AudioTrack | 正常播放→释放 |
| 模型大小 | 35MB + 50MB = **85MB** |

**端侧 TTS 管线完整跑通。**

### 实验全览（同条件对比）

| 实验 | 数据 | 训练方式 | pred-all CER | Holdout CER | 关键变量 |
|------|------|---------|-------------|-------------|---------|
| v2 | re-extracted mel, F0 bug, MFA dur | full_e2e | 77.5% | ~99% | 旧管线 baseline |
| v3 | 同上 | phoneme loss | 84.2% | — | − |
| v4 | 同上 | alternating | 95.3% | — | − |
| v5 | 同上 | mulgate | 83.2% | — | − |
| v6 | 同上 | FiLM | 86.5% | — | − |
| v7 | teacher_mel, F0 fixed, MFA dur | classic | 96.9% | — | dur gap 崩溃 |
| v8 | 同 v7 | full_e2e | 56.5% | — | 帧级错位 |
| **v9** | **teacher_mel, F0 fixed, teacher dur** | **classic** | **24.0%** | **21.5%** | **当前生产** |
| v10a | teacher_mel, MAS dur 展开 | MAS align | 95.5% | 98.4% | dur 坍缩（裸 Viterbi 对齐漂移） |
| v10b | teacher_mel, pred dur 展开, MAS target | MAS align | — (killed) | — | 损失帧错位，mel 平台 0.25 |
| v10c | + 先验(仅DP) + fsum, MAS dur 展开 | MAS align | — (killed) | — | 双稳态：3 次运行 fsum 2.6/7.5/13 |
| v10d | + 卷积 aligner + 先验入 softmax + 闸门 | MAS align | 29.3% / 20.7%(×1.25) | 30.2% / 24.6%(×1.25) | 首个稳定 MAS；补偿后 train 达天花板 |
| v9+scale | teacher dur + 推理 dur×1.27 | classic | **20.3%** | **19.4%** | 全局最优，待进生产 ONNX |
| **v10f** | v10d + warmup 4000 + 30k 步 | MAS align | 29.0% / **20.7%**(×1.25) | 28.2% / **22.6%**(×1.25) | **M2 达成**；闸门 step 4000 开(ema=0.066)，零标签 holdout 22.6% |

**M2 终局（2026-07-31）**：MAS 自学对齐 + 补偿后，train 与 v9+scale 平价
（20.7 vs 20.3，均≈天花板 20.2），holdout 差 3.2pp（22.6 vs 19.4）——
这是"零外部时长标签"的当前溢价，且已低于 v9 原始版的 21.5% 附近。
成熟对齐（warmup 4000）从 5.2pp 收到 3.2pp。checkpoint: `v10f_final.pt`。
遗留改进项：checkpoint 文件名加 run 名（v10f 曾覆盖 v10d 权重）；
fsum 的 8 次 CTC 批量化（当前 10.7 step/s → 预计 15+）。

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
| [ADR-0009](../../docs/decisions/ADR-0009-mas-duration-learning.md) | MAS duration learning |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | 已知问题与修复记录 |
| [NAR_INTERFACE_SPEC.md](NAR_INTERFACE_SPEC.md) | NAR 接口规范 |
| `scripts/mas.py` | MAS DP 算法 + AlignmentModule |
| `scripts/eval_mas_alignment.py` | M1: MAS 对齐质量验证 |
| `scripts/nar_train_align.py` | M2: MAS 联合训练 |
| `scripts/eval_v10.py` | v10 CER 评估 |
