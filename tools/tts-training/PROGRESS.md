# TTS Training Pipeline

端侧中文诗歌 TTS 的训练实验与生产部署。

## 当前生产方案

**PaddleSpeech FS2 ONNX + HiFi-GAN ONNX**（ADR-0007）

- CER = 6.8%，确定性推理，无早停/重复
- FS2 (142MB) + HiFi-GAN (50MB) = 192MB，用户通过 ADB 推送
- Kotlin G2P 端侧实现，268 音素词汇表

## 研究实验：��零训练 FS2（已暂停，见 ADR-0008）

### 目标

训练 7.6M 参数的 FastSpeech 2 用于端侧部署，替代 192MB 的 PaddleSpeech 方案。

### 数据管线

```
诗歌文本 → G2P (pypinyin)
  → PaddleSpeech FS2 ONNX → teacher mel
  → PaddleSpeech HiFiGAN ONNX → teacher audio
  → 特征提取 (mel/f0/energy)
  → MFA 对齐 → per-phoneme durations
  → ASR 质量过滤 (FunASR Paraformer)
  → manifest (.jsonl) + features (.npz)
```

### 实验结果

| 实验 | 数据 | 训练方式 | Holdout delta | Train delta |
|------|------|---------|--------------|-------------|
| H1 | 171首 | 完整诗, GT variance | +70.8% | +1.7% |
| H2 | 171首 | 混合窗口(50/25/25) | +69.9% | — |
| D300 | 300首 | 完整诗, GT variance | +69.9% | — |

��心发现：
- 模型学到的是序列级记忆，不是可组合的音素→声学映射
- 训练方式（完整诗 vs 混合窗口）几乎无影响（H1→H2: <1pp）
- 数据规模（171→300）几乎无影���（H1→D300: <1pp）
- 早期 checkpoint（500-3000步）不存在泛化窗口
- mel L1 与 ASR 可懂度脱钩

### 关键 bug 修复

逗号 G2P 映射 bug：`PUNCT_TO_PHONE` 中 `，`(U+FF0C) 被编码为 U+FFFD（mojibake），
导致 702 个逗号全部映射为 `<unk>`(pid=1) 而非正确的 pid=263。已修复并重新生成所有 manifest。

### E2E Predictor 实验

**目的**：去掉 `--gt_variance`，让 duration/pitch/energy predictor 与 acoustic backbone 联合训练，测试 predicted-variance 推理效果。

**训练配置**：D300 data, 24k steps, save_interval=6000, val_interval=100, wandb=`D300_e2e_predictor`

**训练指标**（最终步 24000）：
| 指标 | 值 |
|------|------|
| train mel L1 | 0.255 |
| val mel L1 | 0.757 |
| train loss | 1.66 (mel=0.26, dur=0.07, pitch=0.65, energy=0.68) |

**评估结果**（holdout_20, 4 checkpoints）：

| Step | P2 (GT recon) | GT-var CER | Pred-var CER | Gap    | Dur L1 |
|------|--------------|------------|-------------|--------|--------|
| 6k   | 27.2%        | 99.6%      | 99.2%       | -0.4%  | 7.4    |
| 12k  | 27.2%        | 99.7%      | 99.0%       | -0.7%  | 7.3    |
| 18k  | 27.2%        | 99.9%      | 99.3%       | -0.6%  | 7.4    |
| 24k  | 27.2%        | 99.5%      | 99.0%       | -0.5%  | 7.3    |

**结论：E2E predictor 训练未能改善泛化。**

- GT-variance CER ~100%，与之前 D300 GT-var 实验一致
- Duration predictor 灾难性崩溃：预测总帧数仅为 GT 的约 31%（如 595 -> 186 帧），音频 3x 加速
- Pred-var 与 GT-var 差距可忽略（-0.5%），两者均已接近 100% CER
- ADR-0008 结论再次确认：模型学的是序列级记忆，而非可组合的音素 -> 声学映射

### 下一步方向

1. **预训练初始化**：从 PaddleSpeech FS2 权重 (37.3M, d_model=384) warm-start
2. **更好的 teacher**：用 CosyVoice 3 生成更高质量诗歌音频
3. **放弃泛化**：将目标诗全部放入训练集，只优化已知诗的表现

## 历史方案

### MOSS-TTS-Nano（已归档）

自回归 token 预测方案，因 token absorbing states 归档为 NO-GO。详见 ADR-0006。

### CosyVoice 3 + MOSS-TTS-Nano（早期实验）

51 首诗的 CosyVoice 3 teacher audio → MOSS-Audio-Tokenizer → MOSS-TTS-Nano fine-tune。
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
