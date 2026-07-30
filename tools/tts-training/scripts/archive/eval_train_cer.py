"""Quick train-set CER evaluation for D300fix GT-var 24k."""
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
OUT = ROOT / "output" / "fix_eval"

with open(STATS_PATH) as f:
    stats = json.load(f)

voc_sess = ort.InferenceSession(
    str(HIFIGAN / "hifigan_csmsc.onnx"),
    providers=["CPUExecutionProvider"],
)

ckpt = torch.load(str(ROOT / "checkpoints" / "D300fix_step24000_slim.pt"),
                   map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()

mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
mel_std = np.array(stats["mel_std"], dtype=np.float32)

# Sample 30 poems from train_300
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
        mel_model_norm = model.mel_linear(dec)[0].numpy()

    mel_model_raw = mel_model_norm * mel_std + mel_mean
    T = min(mel_model_norm.shape[0], mel_gt_raw.shape[0])

    # P2 (GT mel reconstruction)
    audio_gt = voc_sess.run(None, {"logmel": mel_gt_raw[:T].astype(np.float32)})[0].flatten()
    wav_gt = str(OUT / f"train_{pid}_gt.wav")
    sf.write(wav_gt, audio_gt, 24000)
    asr2, _ = transcribe(asr, wav_gt)
    cer2, _, _, _, _ = cer_detail(full_text, asr2)

    # P3 (model mel)
    audio_model = voc_sess.run(None, {"logmel": mel_model_raw[:T].astype(np.float32)})[0].flatten()
    wav_model = str(OUT / f"train_{pid}_model.wav")
    sf.write(wav_model, audio_model, 24000)
    asr3, _ = transcribe(asr, wav_model)
    cer3, _, _, _, _ = cer_detail(full_text, asr3)

    delta = cer3 - cer2
    speed = (idx + 1) / max(time.time() - t0, 1)
    if idx % 5 == 0 or idx < 3:
        print(f"  [{idx+1}/30] {pid}: P2={cer2:.0%} P3={cer3:.0%} d={delta:+.0%} | {full_text[:25]}")

    results.append({"poem_id": pid, "cer2": cer2, "cer3": cer3, "delta": delta})

cers2 = [r["cer2"] for r in results]
cers3 = [r["cer3"] for r in results]
deltas = [r["delta"] for r in results]

print(f"\n{'='*60}")
print(f"D300fix GT-var 24k — TRAIN SET (30 samples)")
print(f"{'='*60}")
print(f"  P2 (GT recon):  CER mean={np.mean(cers2):.1%} median={np.median(cers2):.1%}")
print(f"  P3 (model):     CER mean={np.mean(cers3):.1%} median={np.median(cers3):.1%}")
print(f"  Student delta:  mean={np.mean(deltas):+.1%} median={np.median(deltas):+.1%}")
print(f"  P3=100%:        {sum(1 for c in cers3 if c>=0.99)}/{len(cers3)}")
print(f"  P3<15%:         {sum(1 for c in cers3 if c<0.15)}/{len(cers3)}")
print("\nDONE")
