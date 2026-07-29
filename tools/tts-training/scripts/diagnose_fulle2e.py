"""Deep analysis of Full E2E model: why is train pred-var CER still 80.6%?

The full_e2e model was trained with predicted durations, so the
train/inference gap should be closed. Yet CER is still very high.

This script isolates:
1. How good is the mel prediction now (L1 per phoneme)?
2. Is the duration predictor systematically biased?
3. How much does mel L1 correlate with ASR CER?
4. What does the generated audio actually sound like (mel energy profile)?
5. Is the vocoder + ASR pipeline the bottleneck (P2 baseline)?
"""
import json, math, sys, numpy as np, torch, soundfile as sf, onnxruntime as ort
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "paddle_distill_features"
STATS = json.load(open(ROOT / "data" / "train_300_norm_stats.json"))
mel_mean = np.array(STATS["mel_mean"], dtype=np.float32)
mel_std = np.array(STATS["mel_std"], dtype=np.float32)

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
voc_sess = ort.InferenceSession(
    str(HIFIGAN / "hifigan_csmsc.onnx"),
    providers=["CPUExecutionProvider"],
)

ckpt = torch.load(str(ROOT / "checkpoints" / "FullE2E_step24000_slim.pt"),
                   map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()

# phone map
phone_map = {}
with open(ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0" / "phone_id_map.txt",
          encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            phone_map[parts[0]] = int(parts[1])
id2phone = {v: k for k, v in phone_map.items()}

import random
train = [json.loads(l) for l in open(ROOT / "data" / "train_300_manifest.jsonl", encoding="utf-8")]
random.seed(42)
sample = random.sample(train, 30)
sample.sort(key=lambda r: r["mel_len"])

print("Loading ASR...")
asr = load_asr_model()
print("ASR loaded.\n")

results = []

for idx, rec in enumerate(sample):
    pid = rec["poem_id"]
    npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
    mel_gt_raw = npz["mel"].astype(np.float32)
    dur_gt = np.array(rec["durations"], dtype=np.int32)
    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
    full_text = rec["text"]
    mel_norm_gt = (mel_gt_raw - mel_mean) / mel_std

    # Pred-var inference
    with torch.no_grad():
        mel_pred, log_dur, pred_pitch, pred_energy = model.forward(phone_ids)

    mel_pred_norm = mel_pred[0].numpy()
    mel_predvar_raw = mel_pred_norm * mel_std + mel_mean
    pred_durs = np.maximum(np.round(np.exp(log_dur[0].numpy())), 0).astype(int)

    # Mel L1 (at min length)
    T = min(mel_pred_norm.shape[0], mel_norm_gt.shape[0])
    mel_l1 = np.abs(mel_pred_norm[:T] - mel_norm_gt[:T]).mean()

    # Duration analysis
    dur_l1 = np.abs(pred_durs - dur_gt).mean()
    dur_ratio = pred_durs.sum() / max(dur_gt.sum(), 1)

    # Also compute GT-var mel L1 for comparison
    dur_gt_t = torch.tensor([rec["durations"]], dtype=torch.long)
    f0_gt = npz["f0"].astype(np.float32)
    e_gt = npz["energy"].astype(np.float32)
    f0_norm = np.where(f0_gt > 0,
        (np.log(np.maximum(f0_gt, 1)) - STATS["f0_mean"]) / STATS["f0_std"], 0.0).astype(np.float32)
    e_norm = ((e_gt - STATS["energy_mean"]) / STATS["energy_std"]).astype(np.float32)
    pitch_gt = torch.tensor(f0_norm).unsqueeze(0)
    energy_gt_t = torch.tensor(e_norm).unsqueeze(0)

    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        mi = model.length_regulator(x, dur_gt_t)
        T_out = mi.size(1)
        mi = mi + model.pitch_embed(pitch_gt[:, :T_out].unsqueeze(-1)) + \
                 model.energy_embed(energy_gt_t[:, :T_out].unsqueeze(-1))
        mi = model.pos_enc(mi)
        dec = mi
        for layer in model.decoder_layers:
            dec = layer(dec)
        mel_gtvar = model.mel_linear(dec)[0].numpy()

    T_gt = min(mel_gtvar.shape[0], mel_norm_gt.shape[0])
    mel_l1_gtvar = np.abs(mel_gtvar[:T_gt] - mel_norm_gt[:T_gt]).mean()

    # ASR: GT recon
    audio_p2 = voc_sess.run(None, {"logmel": mel_gt_raw[:T_gt].astype(np.float32)})[0].flatten()
    wav_p2 = str(ROOT / "output" / "fulle2e_eval" / f"diag_{pid}_p2.wav")
    sf.write(wav_p2, audio_p2, 24000)
    asr_p2, _ = transcribe(asr, wav_p2)
    cer_p2, _, _, _, _ = cer_detail(full_text, asr_p2)

    # ASR: GT-var model
    mel_gtvar_raw = mel_gtvar * mel_std + mel_mean
    audio_gtvar = voc_sess.run(None, {"logmel": mel_gtvar_raw[:T_gt].astype(np.float32)})[0].flatten()
    wav_gtvar = str(ROOT / "output" / "fulle2e_eval" / f"diag_{pid}_gtvar.wav")
    sf.write(wav_gtvar, audio_gtvar, 24000)
    asr_gtvar, _ = transcribe(asr, wav_gtvar)
    cer_gtvar, _, _, _, _ = cer_detail(full_text, asr_gtvar)

    # ASR: pred-var model
    audio_pred = voc_sess.run(None, {"logmel": mel_predvar_raw.astype(np.float32)})[0].flatten()
    wav_pred = str(ROOT / "output" / "fulle2e_eval" / f"diag_{pid}_pred.wav")
    sf.write(wav_pred, audio_pred, 24000)
    asr_pred, _ = transcribe(asr, wav_pred)
    cer_pred, _, _, _, _ = cer_detail(full_text, asr_pred)

    # Mel energy profile
    gt_frame_energy = mel_gt_raw.mean(axis=1)
    pred_frame_energy = mel_predvar_raw.mean(axis=1)
    gt_silent = (gt_frame_energy < -5).sum() / len(gt_frame_energy)
    pred_silent = (pred_frame_energy < -5).sum() / max(len(pred_frame_energy), 1)

    if idx < 5 or idx % 10 == 0:
        print(f"[{idx+1}/30] {pid}: L1 gtvar={mel_l1_gtvar:.3f} predvar={mel_l1:.3f} | "
              f"CER p2={cer_p2:.0%} gtvar={cer_gtvar:.0%} pred={cer_pred:.0%} | "
              f"dur_ratio={dur_ratio:.2f} | {full_text[:25]}")

    results.append({
        "poem_id": pid,
        "mel_l1_gtvar": mel_l1_gtvar,
        "mel_l1_predvar": mel_l1,
        "cer_p2": cer_p2,
        "cer_gtvar": cer_gtvar,
        "cer_pred": cer_pred,
        "dur_l1": dur_l1,
        "dur_ratio": dur_ratio,
        "gt_silent_ratio": gt_silent,
        "pred_silent_ratio": pred_silent,
        "pred_dur_total": int(pred_durs.sum()),
        "gt_dur_total": int(dur_gt.sum()),
        "full_text": full_text,
    })

# Aggregate
l1_gt = [r["mel_l1_gtvar"] for r in results]
l1_pv = [r["mel_l1_predvar"] for r in results]
c_p2 = [r["cer_p2"] for r in results]
c_gt = [r["cer_gtvar"] for r in results]
c_pv = [r["cer_pred"] for r in results]
dr = [r["dur_ratio"] for r in results]
gs = [r["gt_silent_ratio"] for r in results]
ps = [r["pred_silent_ratio"] for r in results]

print(f"\n{'='*70}")
print(f"FULL E2E TRAIN ANALYSIS (30 samples)")
print(f"{'='*70}")
print(f"\nMel L1:")
print(f"  GT-var:   mean={np.mean(l1_gt):.3f} min={np.min(l1_gt):.3f} max={np.max(l1_gt):.3f}")
print(f"  Pred-var: mean={np.mean(l1_pv):.3f} min={np.min(l1_pv):.3f} max={np.max(l1_pv):.3f}")
print(f"\nCER:")
print(f"  P2 (GT recon):   mean={np.mean(c_p2):.1%} min={np.min(c_p2):.0%} max={np.max(c_p2):.0%}")
print(f"  GT-var model:    mean={np.mean(c_gt):.1%} min={np.min(c_gt):.0%} max={np.max(c_gt):.0%}")
print(f"  Pred-var model:  mean={np.mean(c_pv):.1%} min={np.min(c_pv):.0%} max={np.max(c_pv):.0%}")
print(f"\nDuration:")
print(f"  Ratio (pred/gt): mean={np.mean(dr):.2f} min={np.min(dr):.2f} max={np.max(dr):.2f}")
print(f"\nSilent frame ratio:")
print(f"  GT:   mean={np.mean(gs):.1%}")
print(f"  Pred: mean={np.mean(ps):.1%}")

# Correlation
corr_l1_cer = np.corrcoef(l1_pv, c_pv)[0, 1]
print(f"\nCorrelation mel L1 vs CER (pred-var): {corr_l1_cer:.3f}")

# Best and worst samples
print(f"\nBest pred-var CER samples:")
for r in sorted(results, key=lambda x: x["cer_pred"])[:5]:
    print(f"  {r['poem_id']}: CER={r['cer_pred']:.0%} L1={r['mel_l1_predvar']:.3f} "
          f"dur_ratio={r['dur_ratio']:.2f} | {r['full_text'][:30]}")

print(f"\nWorst pred-var CER samples:")
for r in sorted(results, key=lambda x: -x["cer_pred"])[:5]:
    print(f"  {r['poem_id']}: CER={r['cer_pred']:.0%} L1={r['mel_l1_predvar']:.3f} "
          f"dur_ratio={r['dur_ratio']:.2f} | {r['full_text'][:30]}")

print("\nDONE")
