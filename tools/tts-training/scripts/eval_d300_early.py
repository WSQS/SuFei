"""D300 early checkpoint evaluation: holdout_20 at 500/1000/1500/2000/2500/3000."""
import argparse, json, math, sys, time, numpy as np, torch, soundfile as sf, onnxruntime as ort
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
STATS_PATH = str(ROOT / "data" / "train_300_norm_stats.json")
FEAT_DIR = str(ROOT / "data" / "paddle_distill_features")
OUT = ROOT / "output" / "unified_eval"

with open(STATS_PATH) as f:
    stats = json.load(f)

voc_sess = ort.InferenceSession(
    str(HIFIGAN / "hifigan_csmsc.onnx"),
    providers=["CPUExecutionProvider"],
)

step_map = {
    "500": "D300early_step500_slim.pt",
    "1000": "D300early_step1000_slim.pt",
    "1500": "D300early_step1500_slim.pt",
    "2000": "D300early_step2000_slim.pt",
    "2500": "D300early_step2500_slim.pt",
    "3000": "D300early_step3000_slim.pt",
}

holdout = [json.loads(l) for l in open(ROOT / "data/holdout_20_manifest.jsonl", encoding="utf-8")]
holdout.sort(key=lambda r: r["mel_len"])

print("Loading ASR...")
asr = load_asr_model()
print("ASR loaded.\n")

mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
mel_std = np.array(stats["mel_std"], dtype=np.float32)

all_curves = {}

for label, fname in step_map.items():
    ckpt_path = ROOT / "checkpoints" / fname
    if not ckpt_path.exists():
        print("SKIP %s: not found" % label)
        continue
    try:
        torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except:
        print("SKIP %s: corrupt" % label)
        continue

    print("\n" + "=" * 70)
    print("D300 EARLY HOLDOUT_20 @ %s" % label)
    print("=" * 70)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = OUT / ("D300early_holdout20_%s" % label)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    for idx, rec in enumerate(holdout):
        pid = rec["poem_id"]
        npz_path = Path(FEAT_DIR) / ("%s.npz" % pid)
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
        mel_gt_norm = (mel_gt_raw - mel_mean) / mel_std
        l1 = float(np.abs(mel_model_norm[:T] - mel_gt_norm[:T]).mean())
        full_text = rec["text"]

        audio_gt = voc_sess.run(None, {"logmel": mel_gt_raw[:T].astype(np.float32)})[0].flatten()
        wav_gt = str(out_dir / ("%s_gt.wav" % pid))
        sf.write(wav_gt, audio_gt, 24000)
        asr2, _ = transcribe(asr, wav_gt)
        cer2, _, _, _, _ = cer_detail(full_text, asr2)

        audio_model = voc_sess.run(None, {"logmel": mel_model_raw[:T].astype(np.float32)})[0].flatten()
        wav_model = str(out_dir / ("%s_model.wav" % pid))
        sf.write(wav_model, audio_model, 24000)
        asr3, _ = transcribe(asr, wav_model)
        cer3, _, _, _, _ = cer_detail(full_text, asr3)

        delta = cer3 - cer2
        if idx % 5 == 0 or idx < 3:
            print("  [%d/%d] %s: L1=%.3f P2=%.0f%% P3=%.0f%% d=%+.0f%% | %s" % (
                idx + 1, len(holdout), pid, l1, cer2 * 100, cer3 * 100, delta * 100, full_text[:25]))

        results.append({"poem_id": pid, "cer2": cer2, "cer3": cer3,
                        "student_delta": delta, "mel_l1": l1})

    deltas = [r["student_delta"] for r in results]
    abs_d = [abs(d) for d in deltas]
    l1s = [r["mel_l1"] for r in results]
    cers3 = [r["cer3"] for r in results]

    s = {
        "n": len(results),
        "delta_mean": float(np.mean(deltas)),
        "delta_median": float(np.median(deltas)),
        "abs_delta_le5pp": sum(1 for d in abs_d if d <= 0.05),
        "p3_100pct": sum(1 for c in cers3 if c >= 0.99),
        "mel_l1_mean": float(np.mean(l1s)),
    }
    print("\n  [%s] delta: mean=%+.1f%% median=%+.1f%% |d|<=5pp=%d/%d P3=100%%=%d/%d L1=%.3f" % (
        label, s["delta_mean"] * 100, s["delta_median"] * 100,
        s["abs_delta_le5pp"], s["n"], s["p3_100pct"], s["n"], s["mel_l1_mean"]))

    rpt = {"name": label, "summary": s, "samples": results}
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(rpt, f, ensure_ascii=False, indent=2)
    all_curves[label] = s

print("\n" + "=" * 70)
print("D300 EARLY CURVE: val_mel_l1 + ASR delta")
print("=" * 70)
print("%-6s %-12s %-12s %-10s %-8s" % ("Step", "delta_mean", "delta_median", "P3=100%", "mel_l1"))
print("-" * 50)
for label in ["500", "1000", "1500", "2000", "2500", "3000"]:
    if label in all_curves:
        s = all_curves[label]
        print("%-6s %+.1f%%       %+.1f%%       %d/%d       %.3f" % (
            label, s["delta_mean"] * 100, s["delta_median"] * 100,
            s["p3_100pct"], s["n"], s["mel_l1_mean"]))

print("\nDONE")
