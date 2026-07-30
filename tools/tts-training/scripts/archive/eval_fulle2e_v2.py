"""Evaluate FullE2E v2 (correct durations) on train + holdout."""
import json, math, sys, time, random
import numpy as np, torch, soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
STATS_PATH = ROOT / "data" / "train_300_norm_stats.json"
FEAT_DIR = ROOT / "data" / "paddle_distill_features"
OUT = ROOT / "output" / "fulle2e_v2_eval"
OUT.mkdir(parents=True, exist_ok=True)

with open(STATS_PATH) as f:
    stats = json.load(f)
mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
mel_std = np.array(stats["mel_std"], dtype=np.float32)

voc_sess = ort.InferenceSession(
    str(HIFIGAN / "hifigan_csmsc.onnx"),
    providers=["CPUExecutionProvider"],
)

ckpt = torch.load(str(ROOT / "checkpoints" / "FullE2E_v2_step24000_slim.pt"),
                   map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()

holdout = [json.loads(l) for l in open(ROOT / "data/holdout_20_manifest.jsonl", encoding="utf-8")]
holdout.sort(key=lambda r: r["mel_len"])

train = [json.loads(l) for l in open(ROOT / "data/train_300_manifest.jsonl", encoding="utf-8")]
random.seed(42)
train_sample = random.sample(train, 30)
train_sample.sort(key=lambda r: r["mel_len"])

print("Loading ASR...")
asr = load_asr_model()
print("ASR loaded.\n")

def eval_set(manifest, label):
    results = []
    t0 = time.time()
    for idx, rec in enumerate(manifest):
        pid = rec["poem_id"]
        npz_path = FEAT_DIR / f"{pid}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(str(npz_path))
        dur_gt = np.array(rec["durations"], dtype=np.int32)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
        full_text = rec["text"]

        with torch.no_grad():
            mel_pred, log_dur, pred_pitch, pred_energy = model.forward(phone_ids)

        mel_predvar_norm = mel_pred[0].numpy()
        mel_predvar_raw = mel_predvar_norm * mel_std + mel_mean
        T_pred = mel_predvar_raw.shape[0]

        pred_durations = np.maximum(np.round(np.exp(log_dur[0].numpy())), 0).astype(int)
        dur_l1 = float(np.abs(pred_durations - dur_gt).mean())

        audio = voc_sess.run(None, {"logmel": mel_predvar_raw[:T_pred].astype(np.float32)})[0].flatten()
        wav_path = str(OUT / f"{pid}_{label}.wav")
        sf.write(wav_path, audio, 24000)
        asr_text, _ = transcribe(asr, wav_path)
        cer, _, _, _, _ = cer_detail(full_text, asr_text)

        if idx < 3 or idx % 10 == 0:
            print(f"  [{idx+1}/{len(manifest)}] {pid}: CER={cer:.0%} dur={dur_gt.sum()}->{pred_durations.sum()} | {full_text[:25]}")

        results.append({"pid": pid, "cer": cer, "dur_l1": dur_l1,
                        "pred_dur": int(pred_durations.sum()), "gt_dur": int(dur_gt.sum())})

    cers = [r["cer"] for r in results]
    print(f"\n  [{label}] ({len(results)} samples):")
    print(f"    Mean CER: {np.mean(cers):.1%}")
    print(f"    100% CER: {sum(1 for c in cers if c >= 0.99)}/{len(cers)}")
    print(f"    <15% CER: {sum(1 for c in cers if c < 0.15)}/{len(cers)}")
    print(f"    Dur L1: {np.mean([r['dur_l1'] for r in results]):.1f}")
    return results

print(f"{'='*70}\nFULL E2E V2 @ 24k (correct durations)\n{'='*70}")
r_train = eval_set(train_sample, "train")
r_holdout = eval_set(holdout, "holdout")
print("\nDONE")
