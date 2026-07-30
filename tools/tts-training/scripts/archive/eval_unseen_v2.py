"""Re-evaluate unseen poems with mel L1 + direct ASR check."""
import json, math, sys, time, numpy as np, torch, soundfile as sf, onnxruntime as ort
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe, normalize_text, align
sys.path.insert(0, str(ROOT / "scripts"))
from eval_unified import cer_detail

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
OUT = ROOT / "output" / "unified_eval" / "E3_unseen_v2"
OUT.mkdir(parents=True, exist_ok=True)

# Load stats
with open(ROOT / "data/paddle_distill_norm_stats.json") as f:
    stats = json.load(f)
mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
mel_std = np.array(stats["mel_std"], dtype=np.float32)

# Load vocoder
voc_sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))

# Load model
ckpt = torch.load(str(ROOT / "checkpoints/E3_decmask_24k.pt"), map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()

# Load ASR
print("Loading ASR...")
asr = load_asr_model()

# Load unseen manifest
manifests = [json.loads(l) for l in open(ROOT / "data/unseen_manifest.jsonl", encoding="utf-8")]
manifests.sort(key=lambda r: r["mel_len"])

results = []
for idx, rec in enumerate(manifests):
    pid = rec["poem_id"]
    npz = np.load(str(ROOT / f"data/unseen_features/{pid}.npz"))
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

    # A0 inference
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

    # Mel L1 (normalized space)
    mel_gt_norm = (mel_gt_raw - mel_mean) / mel_std
    l1 = np.abs(mel_model_norm[:T] - mel_gt_norm[:T]).mean()

    # P2: GT raw mel -> HiFiGAN -> ASR
    audio_gt = voc_sess.run(None, {"logmel": mel_gt_raw[:T].astype(np.float32)})[0].flatten()
    wav_gt = str(OUT / f"{pid}_gt.wav")
    sf.write(wav_gt, audio_gt, 24000)
    asr_text_2, _ = transcribe(asr, wav_gt)

    # P3: Model raw mel (denorm) -> HiFiGAN -> ASR
    audio_model = voc_sess.run(None, {"logmel": mel_model_raw[:T].astype(np.float32)})[0].flatten()
    wav_model = str(OUT / f"{pid}_model.wav")
    sf.write(wav_model, audio_model, 24000)
    asr_text_3, _ = transcribe(asr, wav_model)

    full_text = rec["text"]
    cer2, _, _, _, _ = cer_detail(full_text, asr_text_2)
    cer3, _, _, _, _ = cer_detail(full_text, asr_text_3)
    student_delta = cer3 - cer2

    print(f"[{idx+1}/{len(manifests)}] {pid}: L1={l1:.4f} P2={cer2:.0%} P3={cer3:.0%} delta={student_delta:+.0%}")
    print(f"  ref:  {full_text[:50]}")
    print(f"  asr2: {asr_text_2[:50]}")
    print(f"  asr3: {asr_text_3[:50]}")

    results.append({
        "poem_id": pid, "cer2": cer2, "cer3": cer3,
        "student_delta": student_delta, "mel_l1": float(l1),
        "full_text": full_text, "asr2": asr_text_2, "asr3": asr_text_3,
        "mel_len": rec["mel_len"],
    })

# Summary
cers2 = [r["cer2"] for r in results]
cers3 = [r["cer3"] for r in results]
deltas = [r["student_delta"] for r in results]
abs_deltas = [abs(d) for d in deltas]
print(f"\nSummary: n={len(results)}")
print(f"  P2 mean = {np.mean(cers2):.4f}")
print(f"  P3 mean = {np.mean(cers3):.4f}")
print(f"  student_delta mean = {np.mean(deltas):+.4f}")
print(f"  student_delta median = {np.median(deltas):+.4f}")
print(f"  |delta| <= 5pp: {sum(1 for d in abs_deltas if d <= 0.05)}/{len(results)}")

report = {
    "name": "E3_unseen_v2",
    "n_samples": len(results),
    "summary": {
        "cer2_mean": float(np.mean(cers2)),
        "cer3_mean": float(np.mean(cers3)),
        "student_delta_mean": float(np.mean(deltas)),
        "student_delta_median": float(np.median(deltas)),
    },
    "samples": results,
}
with open(OUT / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\nReport saved to {OUT / 'report.json'}")
