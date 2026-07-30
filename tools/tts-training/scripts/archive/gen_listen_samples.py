"""Generate a few train + holdout samples from D300 24k for listening."""
import json, math, sys, numpy as np, torch, soundfile as sf, onnxruntime as ort
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
STATS_PATH = str(ROOT / "data" / "train_300_norm_stats.json")
FEAT_DIR = str(ROOT / "data" / "paddle_distill_features")
OUT = ROOT / "output" / "listen"

with open(STATS_PATH) as f:
    stats = json.load(f)

voc_sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"), providers=["CPUExecutionProvider"])

ckpt = torch.load(str(ROOT / "checkpoints/D300_step24000_slim.pt"), map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()

mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
mel_std = np.array(stats["mel_std"], dtype=np.float32)

OUT.mkdir(parents=True, exist_ok=True)

# Pick 3 train + 3 holdout
train_picks = ["poem_0001", "poem_0007", "poem_0010"]  # in train_300
holdout_picks = ["poem_0268", "poem_0000", "poem_0204"]  # holdout_20

all_manifests = {}
for mf in ["train_300_manifest.jsonl", "holdout_20_manifest.jsonl"]:
    for line in open(ROOT / "data" / mf, encoding="utf-8"):
        r = json.loads(line)
        all_manifests[r["poem_id"]] = r

for group, picks in [("train", train_picks), ("holdout", holdout_picks)]:
    for pid in picks:
        rec = all_manifests.get(pid)
        if not rec:
            print("SKIP %s: not found" % pid)
            continue
        npz = np.load(str(Path(FEAT_DIR) / ("%s.npz" % pid)))
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
        T = min(mel_model_raw.shape[0], mel_gt_raw.shape[0])

        # Save model + GT
        audio_model = voc_sess.run(None, {"logmel": mel_model_raw[:T].astype(np.float32)})[0].flatten()
        sf.write(str(OUT / ("%s_%s_model.wav" % (group, pid))), audio_model, 24000)

        audio_gt = voc_sess.run(None, {"logmel": mel_gt_raw[:T].astype(np.float32)})[0].flatten()
        sf.write(str(OUT / ("%s_%s_gt.wav" % (group, pid))), audio_gt, 24000)

        print("%s/%s: %s" % (group, pid, rec["text"][:40]))

print("\nDone. Files in: %s" % OUT)
