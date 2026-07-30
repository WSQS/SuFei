"""Evaluate D300fix GT-var model train CER with GT inputs.

This tests the user's hypothesis: if decoder trains with GT pitch/energy,
can it produce good mel on training data?

We test 3 conditions:
  A. GT dur + GT pitch/energy (exact training condition — teacher forcing)
  B. GT dur + predicted pitch/energy (predictor quality on GT dur)
  C. Pred dur + predicted pitch/energy (full inference, same as E2E eval)
"""
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
OUT = ROOT / "output" / "d300fix_gtvar_eval"
OUT.mkdir(parents=True, exist_ok=True)

with open(STATS_PATH) as f:
    stats = json.load(f)
mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
mel_std = np.array(stats["mel_std"], dtype=np.float32)
f0_mean, f0_std = stats["f0_mean"], stats["f0_std"]
e_mean, e_std = stats["energy_mean"], stats["energy_std"]

voc_sess = ort.InferenceSession(
    str(HIFIGAN / "hifigan_csmsc.onnx"),
    providers=["CPUExecutionProvider"],
)

ckpt = torch.load(str(ROOT / "checkpoints" / "D300fix_step24000_slim.pt"),
                   map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded D300fix GT-var checkpoint, step={ckpt.get('step', '?')}")

train = [json.loads(l) for l in open(ROOT / "data/train_300_manifest.jsonl", encoding="utf-8")]
random.seed(42)
train_sample = random.sample(train, 30)
train_sample.sort(key=lambda r: r["mel_len"])

print("Loading ASR...")
asr = load_asr_model()
print("ASR loaded.\n")


def length_regulate_single(x, durations):
    """Expand phoneme-level to frame-level for a single sequence."""
    output = []
    for i, d in enumerate(durations):
        d = max(int(d), 0)
        for _ in range(d):
            output.append(x[i])
    if len(output) == 0:
        output.append(x[0])
    return np.array(output, dtype=np.float32)


def eval_condition(manifest, condition, label):
    """Evaluate CER under specified input condition."""
    results = []
    for idx, rec in enumerate(manifest):
        pid = rec["poem_id"]
        npz_path = FEAT_DIR / f"{pid}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(str(npz_path))
        dur_gt = np.array(rec["durations"], dtype=np.int32)
        f0_gt = npz["f0"].astype(np.float32)
        energy_gt = npz["energy"].astype(np.float32)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
        full_text = rec["text"]
        mel_len = rec["mel_len"]

        f0_norm = np.where(f0_gt > 0,
            (np.log(np.maximum(f0_gt, 1)) - f0_mean) / f0_std, 0.0).astype(np.float32)
        e_norm = ((energy_gt - e_mean) / e_std).astype(np.float32)

        with torch.no_grad():
            # Encoder
            x = model.embedding(phone_ids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x)

            log_dur_pred = model.duration_predictor(x)
            pitch_pred = model.pitch_predictor(x)
            energy_pred = model.energy_predictor(x)

            if condition == "A":
                # GT dur + GT pitch/energy
                durations = torch.tensor(dur_gt).unsqueeze(0)
                pitches = torch.tensor(f0_norm[:mel_len]).unsqueeze(0)
                energies = torch.tensor(e_norm[:mel_len]).unsqueeze(0)
                mel_out, _, _, _ = model.forward(
                    phone_ids, durations=durations,
                    pitches=pitches, energies=energies)
            elif condition == "B":
                # GT dur + predicted pitch/energy
                durations = torch.tensor(dur_gt).unsqueeze(0)
                mel_out, _, _, _ = model.forward(
                    phone_ids, durations=durations,
                    pitches=None, energies=None)
            elif condition == "C":
                # Predicted everything
                mel_out, _, _, _ = model.forward(phone_ids)

        mel_pred_norm = mel_out[0].numpy()
        mel_pred_raw = mel_pred_norm * mel_std + mel_mean
        T = mel_pred_raw.shape[0]

        audio = voc_sess.run(None, {"logmel": mel_pred_raw[:T].astype(np.float32)})[0].flatten()
        wav_path = str(OUT / f"{pid}_{label}.wav")
        sf.write(wav_path, audio, 24000)
        asr_text, _ = transcribe(asr, wav_path)
        cer, _, _, _, _ = cer_detail(full_text, asr_text)

        if idx < 3 or idx % 10 == 0:
            print(f"  [{idx+1}/{len(manifest)}] {pid}: CER={cer:.0%} | {full_text[:25]}")

        results.append({"pid": pid, "cer": cer})

    cers = [r["cer"] for r in results]
    print(f"\n  [{label}] ({len(results)} samples):")
    print(f"    Mean CER: {np.mean(cers):.1%}")
    print(f"    100% CER: {sum(1 for c in cers if c >= 0.99)}/{len(cers)}")
    print(f"    <15% CER: {sum(1 for c in cers if c < 0.15)}/{len(cers)}")
    return results


print(f"{'='*70}")
print("D300fix GT-var @ 24k — Train CER Decomposition (30 samples)")
print(f"{'='*70}")

print("\n--- Condition A: GT dur + GT pitch/energy (teacher forcing) ---")
rA = eval_condition(train_sample, "A", "gtvar_A")

print("\n--- Condition B: GT dur + pred pitch/energy ---")
rB = eval_condition(train_sample, "B", "gtvar_B")

print("\n--- Condition C: pred dur + pred pitch/energy (full inference) ---")
rC = eval_condition(train_sample, "C", "gtvar_C")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  A (GT all):     CER = {np.mean([r['cer'] for r in rA]):.1%}")
print(f"  B (GT dur):     CER = {np.mean([r['cer'] for r in rB]):.1%}")
print(f"  C (pred all):   CER = {np.mean([r['cer'] for r in rC]):.1%}")
print(f"\nFor reference, Full E2E v2 train CER = 77.5%")
print("DONE")
