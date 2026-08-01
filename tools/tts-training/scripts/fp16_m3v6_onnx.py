"""fp16-quantize m3_v6 FS2 + HiFiGAN ONNX and re-verify CER on the same protocol.

keep_io_types=True keeps every model input/output in fp32 (Cast nodes at the
boundaries), so the numpy denorm glue and HiFiGAN feed are unchanged; only the
internal weights/activations become fp16 -> ~2x smaller files.

Two configs, same holdout-20 / train-30 (seed 42), to attribute any drop:
  A = FS2 fp16 + HiFiGAN fp32   (safe fallback if vocoder is fp16-sensitive)
  B = FS2 fp16 + HiFiGAN fp16   (smallest package)
fp32 baseline (from export step): train 20.6% / holdout 21.0%.
"""
import json, sys, random, os
import numpy as np
import onnx
from onnxconverter_common import float16
import onnxruntime as ort
import soundfile as sf
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

OUTDIR = ROOT / 'models' / 'sufei_fs2_onnx_m3v6'
FS2_FP32 = OUTDIR / 'fastspeech2_sufei_m3v6.onnx'
FS2_FP16 = OUTDIR / 'fastspeech2_sufei_m3v6_fp16.onnx'
HIFI_FP32 = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0' / 'hifigan_csmsc.onnx'
HIFI_FP16 = OUTDIR / 'hifigan_csmsc_fp16.onnx'
STATS_PATH = ROOT / 'data' / 'm3p2_norm_stats.json'
HOLDOUT_MANIFEST = ROOT / 'data' / 'm3p2b_holdout_manifest.jsonl'
TRAIN_MANIFEST = ROOT / 'data' / 'm3b_train_manifest.jsonl'


# repeat_interleave in the length regulator exports to SplitToSequence /
# ConcatFromSequence (sequence-typed tensors). onnxconverter_common does not
# propagate fp16 into sequence element types, so both the producer and consumer
# of every sequence must stay fp32 (converter auto-casts at the boundaries).
# These ops carry no weights -> keeping them fp32 costs no model size.
SEQ_OPS = ['SplitToSequence', 'ConcatFromSequence', 'SequenceAt', 'SequenceConstruct',
           'SequenceEmpty', 'SequenceInsert', 'SequenceErase', 'SequenceLength',
           'SequenceMap', 'Loop', 'Scan']
_DEFAULT_BLOCK = list(getattr(float16, 'DEFAULT_OP_BLOCK_LIST', []))
BLOCK = list(dict.fromkeys(_DEFAULT_BLOCK + SEQ_OPS))


def to_fp16(src, dst, label):
    m = onnx.load(str(src))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True, op_block_list=BLOCK)
    onnx.save(m16, str(dst))
    # fail fast if the converted graph is not loadable
    ort.InferenceSession(str(dst), providers=['CPUExecutionProvider'])
    s0, s1 = os.path.getsize(src) / 1e6, os.path.getsize(dst) / 1e6
    print(f"{label}: {s0:.1f} MB -> {s1:.1f} MB fp16 ({dst.name}) [load OK]")


to_fp16(FS2_FP32, FS2_FP16, "FS2")
to_fp16(HIFI_FP32, HIFI_FP16, "HiFiGAN")

stats = json.load(open(STATS_PATH))
mel_mean = np.array(stats['mel_mean'], dtype=np.float32)
mel_std = np.array(stats['mel_std'], dtype=np.float32)

prov = ['CPUExecutionProvider']
fs2_32 = ort.InferenceSession(str(FS2_FP32), providers=prov)
fs2_16 = ort.InferenceSession(str(FS2_FP16), providers=prov)
hifi_32 = ort.InferenceSession(str(HIFI_FP32), providers=prov)
hifi_16 = ort.InferenceSession(str(HIFI_FP16), providers=prov)

asr = load_asr_model()

holdout = [json.loads(l) for l in open(HOLDOUT_MANIFEST, encoding='utf-8')]
train = [json.loads(l) for l in open(TRAIN_MANIFEST, encoding='utf-8')]
random.seed(42)
train_sample = random.sample(train, 30)


def mel_of(sess, phone_ids):
    return sess.run(None, {"text": phone_ids})[0]


def audio_of(voc, mel_norm):
    mel_raw = mel_norm * mel_std + mel_mean
    return voc.run(None, {'logmel': mel_raw.astype(np.float32)})[0].flatten()


def run_config(records, fs2, voc, prefix):
    cers = []
    for rec in records:
        phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)
        audio = audio_of(voc, mel_of(fs2, phone_ids))
        wav = str(OUTDIR / f"{prefix}{rec['poem_id']}.wav")
        sf.write(wav, audio, 24000)
        asr_text, _ = transcribe(asr, wav)
        cer, _, _, _, _ = cer_detail(rec['text'], asr_text)
        cers.append(cer)
    return np.array(cers)


# numeric mel drift fp32 vs fp16 FS2 (no ASR, fast sanity)
mel_mae = []
for rec in train_sample[:10]:
    pid_ids = np.array(rec['phoneme_ids'], dtype=np.int64)
    m32, m16 = mel_of(fs2_32, pid_ids), mel_of(fs2_16, pid_ids)
    if m32.shape == m16.shape:
        mel_mae.append(np.abs(m32 - m16).mean())
    else:
        mel_mae.append(float('nan'))
print(f"\nFS2 fp16 mel MAE vs fp32 (10 train, same-shape only): "
      f"mean={np.nanmean(mel_mae):.4f}  shape-mismatch={sum(np.isnan(mel_mae))}/10")

print("\n=== Config A: FS2 fp16 + HiFiGAN fp32 ===")
a_h = run_config(holdout, fs2_16, hifi_32, 'A_holdout_')
a_t = run_config(train_sample, fs2_16, hifi_32, 'A_train_')
print("  Holdout %.1f%%  Train %.1f%%" % (a_h.mean()*100, a_t.mean()*100))

print("\n=== Config B: FS2 fp16 + HiFiGAN fp16 ===")
b_h = run_config(holdout, fs2_16, hifi_16, 'B_holdout_')
b_t = run_config(train_sample, fs2_16, hifi_16, 'B_train_')
print("  Holdout %.1f%%  Train %.1f%%" % (b_h.mean()*100, b_t.mean()*100))

print("\n" + "=" * 64)
print("  m3_v6 fp16 parity (fp32 baseline: train 20.6% / holdout 21.0%)")
print("  Config A (FS2-16 + HiFi-32): train %.1f%% (%+.1f)  holdout %.1f%% (%+.1f)" % (
    a_t.mean()*100, a_t.mean()*100 - 20.6, a_h.mean()*100, a_h.mean()*100 - 21.0))
print("  Config B (FS2-16 + HiFi-16): train %.1f%% (%+.1f)  holdout %.1f%% (%+.1f)" % (
    b_t.mean()*100, b_t.mean()*100 - 20.6, b_h.mean()*100, b_h.mean()*100 - 21.0))
sizeA = (os.path.getsize(FS2_FP16) + os.path.getsize(HIFI_FP32)) / 1e6
sizeB = (os.path.getsize(FS2_FP16) + os.path.getsize(HIFI_FP16)) / 1e6
print("  Package size  A: %.1f MB   B: %.1f MB   (fp32 was 87.6 MB)" % (sizeA, sizeB))
print("=" * 64)
print("DONE")
