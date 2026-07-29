"""Quick train-set CER evaluation for D300fix E2E 24k (predicted variance).

Tests both GT-var and Pred-var on the same 30 train samples for direct comparison.
"""
import json, math, sys, time, numpy as np, torch, soundfile as sf, onnxruntime as ort
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
STATS_PATH = ROOT / "data" / "train_300_norm_stats.json"
FEAT_DIR = ROOT / "data" / "paddle_distill_features"
OUT = ROOT / "output" / "fixe2e_train_eval"
OUT.mkdir(parents=True, exist_ok=True)

with open(STATS_PATH) as f:
    stats = json.load(f)

voc_sess = ort.InferenceSession(
    str(HIFIGAN / "hifigan_csmsc.onnx"),
    providers=["CPUExecutionProvider"],
)

ckpt = torch.load(str(ROOT / "checkpoints" / "D300fixE2E_step24000_slim.pt"),
                   map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()

mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
mel_std = np.array(stats["mel_std"], dtype=np.float32)

train = [json.loads(l) for l in open(ROOT / "data" / "train_300_manifest.jsonl", encoding="utf-8")]
import random
random.seed(42)
sample = random.sample(train, 30)
sample.sort(key=lambda r: r["mel_len"])

print("Loading ASR...")
asr = load_asr_model()
print("ASR loaded.\n")

results = []
t0 = time.time()

for idx, rec in enumerate(sample):
    pid = rec["poem_id"]
    npz_path = FEAT_DIR / f"{pid}.npz"
    if not npz_path.exists():
        continue
    npz = np.load(str(npz_path))
    mel_gt_raw = npz["mel"].astype(np.float32)
    f0_gt = npz["f0"].astype(np.float32)
    e_gt = npz["energy"].astype(np.float32)
    dur_gt = np.array(rec["durations"], dtype=np.int32)
    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
    dur_gt_t = torch.tensor(np.array([dur_gt]), dtype=torch.long)
    f0_norm = np.where(f0_gt > 0, (np.log(np.maximum(f0_gt, 1)) - stats["f0_mean"]) / stats["f0_std"], 0.0).astype(np.float32)
    e_norm = ((e_gt - stats["energy_mean"]) / stats["energy_std"]).astype(np.float32)
    pitch_gt = torch.tensor(f0_norm).unsqueeze(0)
    energy_gt_t = torch.tensor(e_norm).unsqueeze(0)
    full_text = rec["text"]

    # --- GT variance path ---
    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        mel_input = model.length_regulator(x, dur_gt_t)
        T_out = mel_input.size(1)
        mel_input = mel_input + \
            model.pitch_embed(pitch_gt[:, :T_out].unsqueeze(-1)) + \
            model.energy_embed(energy_gt_t[:, :T_out].unsqueeze(-1))
        mel_input = model.pos_enc(mel_input)
        dec = mel_input
        for layer in model.decoder_layers:
            dec = layer(dec)
        mel_gtvar = model.mel_linear(dec)[0].numpy()

    mel_gtvar_raw = mel_gtvar * mel_std + mel_mean
    T_gt = min(mel_gtvar.shape[0], mel_gt_raw.shape[0])

    audio_gtvar = voc_sess.run(None, {"logmel": mel_gtvar_raw[:T_gt].astype(np.float32)})[0].flatten()
    wav_gtvar = str(OUT / f"{pid}_gtvar.wav")
    sf.write(wav_gtvar, audio_gtvar, 24000)
    asr_gtvar, _ = transcribe(asr, wav_gtvar)
    cer_gtvar, _, _, _, _ = cer_detail(full_text, asr_gtvar)

    # --- Predicted variance path ---
    with torch.no_grad():
        mel_pred, log_dur, pred_pitch, pred_energy = model.forward(phone_ids)

    mel_predvar = mel_pred[0].numpy()
    mel_predvar_raw = mel_predvar * mel_std + mel_mean
    T_pred = mel_predvar.shape[0]

    audio_predvar = voc_sess.run(None, {"logmel": mel_predvar_raw[:T_pred].astype(np.float32)})[0].flatten()
    wav_predvar = str(OUT / f"{pid}_predvar.wav")
    sf.write(wav_predvar, audio_predvar, 24000)
    asr_predvar, _ = transcribe(asr, wav_predvar)
    cer_predvar, _, _, _, _ = cer_detail(full_text, asr_predvar)

    # Duration stats
    pred_durations = np.maximum(np.round(np.exp(log_dur[0].numpy())), 0).astype(int)
    dur_l1 = float(np.abs(pred_durations - dur_gt).mean())

    speed = (idx + 1) / max(time.time() - t0, 1)
    if idx % 5 == 0 or idx < 3:
        print(f"  [{idx+1}/30] {pid}: GTvar={cer_gtvar:.0%} PredVar={cer_predvar:.0%} "
              f"gap={cer_predvar-cer_gtvar:+.0%} durL1={dur_l1:.1f} | {full_text[:25]}")

    results.append({"poem_id": pid, "cer_gtvar": cer_gtvar, "cer_predvar": cer_predvar,
                    "gap": cer_predvar - cer_gtvar, "dur_l1": dur_l1})

cers_gt = [r["cer_gtvar"] for r in results]
cers_pv = [r["cer_predvar"] for r in results]
gaps = [r["gap"] for r in results]
dur_l1s = [r["dur_l1"] for r in results]

print(f"\n{'='*60}")
print(f"D300fix E2E 24k — TRAIN SET (30 samples)")
print(f"{'='*60}")
print(f"  GT-var:    CER mean={np.mean(cers_gt):.1%} median={np.median(cers_gt):.1%}  100%={sum(1 for c in cers_gt if c>=0.99)}/30  <15%={sum(1 for c in cers_gt if c<0.15)}/30")
print(f"  Pred-var:  CER mean={np.mean(cers_pv):.1%} median={np.median(cers_pv):.1%}  100%={sum(1 for c in cers_pv if c>=0.99)}/30  <15%={sum(1 for c in cers_pv if c<0.15)}/30")
print(f"  Gap:       mean={np.mean(gaps):+.1%} median={np.median(gaps):+.1%}")
print(f"  Dur L1:    mean={np.mean(dur_l1s):.1f}")
print("\nDONE")
