# MOSS-TTS-Nano TTS 诊断报告

## 版本
- **最终模型**: moss_poetry_sft_320_v4 (max_length=8192, 3 epochs, 320 samples)
- **测试集版本**: test_set_v1 (24 poems, frozen 2026-07-26)
- **ASR 基线**: FunASR Paraformer-large, teacher median CER=14.3%, P95=23.2%

## 训练演进

| 版本 | max_length | 截断率 | EOS 保留 | 关键修复 |
|------|-----------|--------|---------|---------|
| v1 (51 samples) | 1024 | ~100% | 0% | 初始实验 |
| v2 (320 samples) | 2048 | 99.1% | 0% | 数据对齐（标题+作者） |
| v3 (320 samples) | 4096 | ~53% | ~47% | 部分截断 |
| **v4** | **8192** | **0%** | **100%** | **动态 padding + 完整 EOS** |

## v4 评估结果

### Sampling (do_sample=True, repetition_penalty=1.2)
- **PASS: 12/24 (50%)** — CER 多为 0-9%
- **WARN: 2/24 (8.3%)** — 高 CER 但完整
- **FAIL: 10/24 (41.7%)** — 早停 + 缺行

### Greedy (do_sample=False)
- **FAIL: 24/24 (100%)** — ASR 无可识别内容
- 所有音频生成满 750 帧被硬截断

## 结论

MOSS v4 在 sampling 模式下能够为部分样本生成高内容准确度的语音（多数 PASS 样本
CER < 9%），说明模型已具备一定的文本到语音建模能力。但其推理稳定性无法满足
产品要求：固定测试集中仅 50% 样本达到 PASS，且存在随机早停；greedy decoding
在全部测试样本上均退化为不可识别音频，并在多个 audio codebook 中进入长期重复的
token 吸收态。

进一步的 teacher-forced 和 EOS 诊断发现了 token ID、head 或 channel 映射疑点
（top1=931 vs gt=9 处于不同编号空间），因此目前无法确定根因属于训练目标错误、
概率校准、解码状态机还是诊断实现。但这些未决问题不影响产品决策：继续���复将增加
训练、解码和 Android 部署复杂度，且仍不能获得显式 duration 控制。

**决定：MOSS v4 作为实验性 AR 基线归档，不再作为产品主线。
后续产品研发转向具有显式时长建模的 NAR 架构。**

## 故障取证结果

### 已确认
- Greedy 音频不是正常语音（RMS 极低，83% 静音，高频噪声）
- Greedy 多个 codebook token 进入吸收态（连续重复 530+ 帧）
- Sampling 产品成功率 50%（12 PASS / 2 WARN / 10 FAIL）
- Deterministic decoding 完全不可用

### 未确认（存在诊断实现疑点）
- Teacher-forced argmax 准确率 0.5% — top1=931 vs gt=9 疑似 ID 空间不匹配
- EOS 排名 16384 — 可能查询了错误的 head 或 vocabulary
- 待查 issue: audio-head label/logit alignment

## 归档清单
- Checkpoint: output/moss_poetry_sft_320_v4/checkpoint-epoch-3
- 测试集: data/test_set_v1.json
- 评估脚本: scripts/eval_tts.py
- 评估结果: data/eval_v4.csv, data/eval_v4_greedy.csv
- 生成音频: output/test_audio_v4/, output/test_audio_v4_greedy/
- wandb: https://wandb.ai/sophomore42/sufei-tts-finetune/runs/zxhfqbwl
- 训练日志: data/train_v4_wandb.log
