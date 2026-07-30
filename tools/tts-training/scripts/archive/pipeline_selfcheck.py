"""Pipeline self-check: reprocess a known training poem with the unseen pipeline,
then compare old vs new features and run E3 A0 on both.

This excludes feature pipeline drift as the cause of unseen set failure.

Steps:
  1. Pick a training poem (short, well-fitted: e.g. poem_0285)
  2. Re-run the full unseen pipeline on its text (FS2 → HiFiGAN → features → MFA)
  3. Compare old .npz vs new .npz (mel/f0/energy shapes, stats, L1)
  4. Compare phoneme_ids (must be identical)
  5. Compare duration sum == mel_len
  6. Run E3 A0 on both old and new features, compare mel L1
"""
import json, math, sys, subprocess, numpy as np, torch, soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
sys.path.insert(0, str(ROOT / "scripts"))

from generate_paddlespeech_distillation_data import (
    text_to_phonemes, phones_to_ids, load_phone_id_map,
    run_fs2, run_hifigan, extract_mel, extract_f0, extract_energy,
    parse_textgrid_intervals, build_durations, FRAME_RATE, SR,
)
from nar_fastspeech2 import FastSpeech2

FS2_DIR = ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0"
HIFIGAN_DIR = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"

OUT = ROOT / "output" / "pipeline_selfcheck"
OUT.mkdir(parents=True, exist_ok=True)

# Pick a well-fitted short training poem
TEST_PID = "poem_0285"  # mel_len=552, one of the shortest

# Load training manifest to get the text
with open(ROOT / "data/paddle_distill_manifest.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["poem_id"] == TEST_PID:
            train_rec = r
            break

text = train_rec["text"]
print(f"Test poem: {TEST_PID}")
print(f"  text: {text[:60]}")
print(f"  train mel_len: {train_rec['mel_len']}")
print(f"  train n_phonemes: {train_rec['n_phonemes']}")

# ── Step 1: Compare phoneme_ids ──
print(f"\n{'='*60}")
print("Step 1: Phoneme ID comparison")
print(f"{'='*60}")

phone_map = load_phone_id_map()
phonemes = text_to_phonemes(text)
new_ids, unmapped = phones_to_ids(phonemes, phone_map)
old_ids = train_rec["phoneme_ids"]

print(f"  Old phoneme_ids ({len(old_ids)}): {old_ids[:20]}...")
print(f"  New phoneme_ids ({len(new_ids)}): {new_ids[:20]}...")
if unmapped:
    print(f"  Unmapped: {unmapped}")
ids_match = old_ids == new_ids
print(f"  IDs identical: {ids_match}")
if not ids_match:
    for i, (a, b) in enumerate(zip(old_ids, new_ids)):
        if a != b:
            print(f"  First diff at pos {i}: old={a} new={b}")
    if len(old_ids) != len(new_ids):
        print(f"  Length diff: old={len(old_ids)} new={len(new_ids)}")

# ── Step 2: Regenerate features with unseen pipeline ──
print(f"\n{'='*60}")
print("Step 2: Regenerate features (FS2 → HiFiGAN → extract)")
print(f"{'='*60}")

fs2_sess = ort.InferenceSession(str(FS2_DIR / "fastspeech2_csmsc.onnx"),
                                providers=["CPUExecutionProvider"])
hifigan_sess = ort.InferenceSession(str(HIFIGAN_DIR / "hifigan_csmsc.onnx"),
                                    providers=["CPUExecutionProvider"])

# FS2 → teacher mel → HiFiGAN → audio
teacher_mel = run_fs2(fs2_sess, new_ids)
audio = run_hifigan(hifigan_sess, teacher_mel)

# Save audio
wav_path = str(OUT / f"{TEST_PID}_regen.wav")
sf.write(wav_path, audio, SR)
print(f"  Audio: {len(audio)/SR:.1f}s")

# Extract features
new_mel = extract_mel(audio, SR)
new_f0 = extract_f0(audio, SR)
new_energy = extract_energy(new_mel)

min_len = min(len(new_f0), new_mel.shape[0], len(new_energy))
new_mel = new_mel[:min_len].astype(np.float32)
new_f0 = new_f0[:min_len].astype(np.float32)
new_energy = new_energy[:min_len].astype(np.float32)

# Save new features
np.savez_compressed(str(OUT / f"{TEST_PID}_regen.npz"),
                    mel=new_mel, f0=new_f0, energy=new_energy)

# ── Step 3: Compare old vs new features ──
print(f"\n{'='*60}")
print("Step 3: Old vs New feature comparison")
print(f"{'='*60}")

old_npz = np.load(str(ROOT / f"data/paddle_distill_features/{TEST_PID}.npz"))
old_mel = old_npz["mel"].astype(np.float32)
old_f0 = old_npz["f0"].astype(np.float32)
old_energy = old_npz["energy"].astype(np.float32)

print(f"  Old mel shape: {old_mel.shape}")
print(f"  New mel shape: {new_mel.shape}")
print(f"  Old f0 shape:  {old_f0.shape}")
print(f"  New f0 shape:  {new_f0.shape}")

# Mel stats comparison
print(f"\n  Mel stats:")
print(f"    Old range: [{old_mel.min():.2f}, {old_mel.max():.2f}]  mean={old_mel.mean():.3f}")
print(f"    New range: [{new_mel.min():.2f}, {new_mel.max():.2f}]  mean={new_mel.mean():.3f}")

# Per-frequency comparison
T = min(old_mel.shape[0], new_mel.shape[0])
mel_diff = np.abs(old_mel[:T] - new_mel[:T])
print(f"    Mel L1 (old vs new, first {T} frames): {mel_diff.mean():.4f}")
print(f"    Mel max diff: {mel_diff.max():.4f}")

# F0 comparison
T_f0 = min(len(old_f0), len(new_f0))
f0_diff = np.abs(old_f0[:T_f0] - new_f0[:T_f0])
print(f"\n  F0 stats:")
print(f"    Old range: [{old_f0.min():.1f}, {old_f0.max():.1f}]")
print(f"    New range: [{new_f0.min():.1f}, {new_f0.max():.1f}]")
print(f"    F0 L1: {f0_diff.mean():.3f}")

# Energy comparison
T_e = min(len(old_energy), len(new_energy))
e_diff = np.abs(old_energy[:T_e] - new_energy[:T_e])
print(f"\n  Energy stats:")
print(f"    Old range: [{old_energy.min():.1f}, {old_energy.max():.1f}]")
print(f"    New range: [{new_energy.min():.1f}, {new_energy.max():.1f}]")
print(f"    Energy L1: {e_diff.mean():.3f}")

# ── Step 4: Check if teacher audio also matches the one in paddle_mfa_corpus ──
print(f"\n{'='*60}")
print("Step 4: Teacher audio comparison")
print(f"{'='*60}")

old_teacher_wav = ROOT / "data" / "paddle_mfa_corpus" / f"{TEST_PID}.wav"
if old_teacher_wav.exists():
    old_audio, old_sr = sf.read(str(old_teacher_wav))
    print(f"  Old teacher wav: {len(old_audio)/old_sr:.2f}s, sr={old_sr}")
    print(f"  Old amp: [{old_audio.min():.3f}, {old_audio.max():.3f}] rms={np.sqrt(np.mean(old_audio**2)):.3f}")
else:
    print(f"  Old teacher wav not found at {old_teacher_wav}")
print(f"  New teacher wav: {len(audio)/SR:.2f}s, sr={SR}")
print(f"  New amp: [{audio.min():.3f}, {audio.max():.3f}] rms={np.sqrt(np.mean(audio**2)):.3f}")

# ── Step 5: Run MFA on regenerated audio to get durations ──
print(f"\n{'='*60}")
print("Step 5: MFA re-alignment")
print(f"{'='*60}")

# Prepare MFA corpus
mfa_corpus = OUT / "mfa_corpus"
mfa_corpus.mkdir(parents=True, exist_ok=True)
mfa_output = OUT / "mfa_aligned"
mfa_output.mkdir(parents=True, exist_ok=True)

# Write 16kHz version for MFA
import librosa as lr
audio_16k = lr.resample(audio, orig_sr=SR, target_sr=16000)
sf.write(str(mfa_corpus / f"{TEST_PID}.wav"), audio_16k, 16000)
(mfa_corpus / f"{TEST_PID}.lab").write_text(text, encoding="utf-8")

dict_path = str(ROOT / "data" / "mfa_char_dictionary.txt")

# Run MFA
import shutil
if mfa_output.exists():
    shutil.rmtree(mfa_output)
mfa_output.mkdir(parents=True)

env = {
    "PATH": r"C:\Users\wsqsy\.conda\envs\mfa\Library\bin;" 
            r"C:\Users\wsqsy\.conda\envs\mfa;"
            r"C:\Users\wsqsy\.conda\envs\mfa\Scripts;" 
            + __import__("os").environ.get("PATH", ""),
}
cmd = [
    r"C:\Users\wsqsy\.conda\envs\mfa\Scripts\mfa.exe",
    "align", str(mfa_corpus), dict_path, "mandarin_mfa", str(mfa_output),
    "--overwrite", "--clean", "--num_jobs", "1",
    "--beam", "100", "--retry_beam", "400",
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
print(f"  MFA return code: {result.returncode}")

tg_path = mfa_output / f"{TEST_PID}.TextGrid"
if tg_path.exists():
    tg_intervals = parse_textgrid_intervals(tg_path)
    new_durations = build_durations(phonemes, tg_intervals)
    
    # Fix duration sum
    if sum(new_durations) != min_len:
        diff = min_len - sum(new_durations)
        max_idx = max(range(len(new_durations)), key=lambda x: new_durations[x])
        new_durations[max_idx] = max(new_durations[max_idx] + diff, 1)
    
    print(f"  New durations: sum={sum(new_durations)}, mel_len={min_len}")
    print(f"  Old durations: sum={sum(train_rec['durations'])}, mel_len={train_rec['mel_len']}")
    
    dur_diff = np.abs(np.array(new_durations[:len(train_rec['durations'])]) - np.array(train_rec['durations']))
    print(f"  Duration L1: {dur_diff.mean():.2f}")
else:
    print(f"  MFA TextGrid not found!")
    new_durations = train_rec["durations"]  # fallback

# ── Step 6: Run E3 A0 on both old and new features ──
print(f"\n{'='*60}")
print("Step 6: E3 A0 comparison (old features vs new features)")
print(f"{'='*60}")

# Load norm stats
with open(ROOT / "data/paddle_distill_norm_stats.json") as f:
    stats = json.load(f)
mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
mel_std = np.array(stats["mel_std"], dtype=np.float32)

# Load model
ckpt = torch.load(str(ROOT / "checkpoints/E3_decmask_24k.pt"), map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()


def run_a0_l1(mel_gt_raw, f0_gt, e_gt, dur_gt, phone_ids, label):
    """Run A0 and return mel L1 in normalized space."""
    f0_norm = np.where(f0_gt > 0,
                       (np.log(np.maximum(f0_gt, 1)) - stats["f0_mean"]) / stats["f0_std"],
                       0.0).astype(np.float32)
    e_norm = ((e_gt - stats["energy_mean"]) / stats["energy_std"]).astype(np.float32)
    
    phone_ids_t = torch.tensor([phone_ids], dtype=torch.long)
    dur_t = torch.tensor(np.array([dur_gt]), dtype=torch.long)
    pitch_t = torch.tensor(f0_norm).unsqueeze(0)
    energy_t = torch.tensor(e_norm).unsqueeze(0)
    
    with torch.no_grad():
        x = model.embedding(phone_ids_t) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        mel_input = model.length_regulator(x, dur_t)
        T_out = mel_input.size(1)
        mel_input = mel_input + \
            model.pitch_embed(pitch_t[:, :T_out].unsqueeze(-1)) + \
            model.energy_embed(energy_t[:, :T_out].unsqueeze(-1))
        mel_input = model.pos_enc(mel_input)
        dec = mel_input
        for layer in model.decoder_layers:
            dec = layer(dec)
        mel_pred_norm = model.mel_linear(dec)[0].numpy()
    
    mel_gt_norm = (mel_gt_raw - mel_mean) / mel_std
    T = min(mel_pred_norm.shape[0], mel_gt_norm.shape[0])
    l1 = np.abs(mel_pred_norm[:T] - mel_gt_norm[:T]).mean()
    
    # Also check pred mel stats
    mel_pred_raw = mel_pred_norm * mel_std + mel_mean
    
    print(f"\n  [{label}]")
    print(f"    mel_pred norm range: [{mel_pred_norm.min():.2f}, {mel_pred_norm.max():.2f}]")
    print(f"    mel_gt norm range:   [{mel_gt_norm.min():.2f}, {mel_gt_norm.max():.2f}]")
    print(f"    mel_pred raw range:  [{mel_pred_raw.min():.2f}, {mel_pred_raw.max():.2f}]")
    print(f"    mel_gt raw range:    [{mel_gt_raw.min():.2f}, {mel_gt_raw.max():.2f}]")
    print(f"    Mel L1 (normalized): {l1:.4f}")
    print(f"    T_pred={mel_pred_norm.shape[0]}, T_gt={mel_gt_norm.shape[0]}")
    return l1


# Old features
l1_old = run_a0_l1(old_mel, old_f0, old_energy, train_rec["durations"], train_rec["phoneme_ids"], "OLD features")

# New features (if durations succeeded)
if tg_path.exists():
    l1_new = run_a0_l1(new_mel, new_f0, new_energy, new_durations, new_ids, "NEW features (regenerated)")
else:
    # Use old durations with new features
    l1_new = run_a0_l1(new_mel, new_f0, new_energy, train_rec["durations"], new_ids, "NEW features (old durs)")

# ── Summary ──
print(f"\n{'='*60}")
print("PIPELINE SELFCHECK SUMMARY")
print(f"{'='*60}")
print(f"  Phoneme IDs identical:     {ids_match}")
print(f"  Old mel_len:               {train_rec['mel_len']}")
print(f"  New mel_len:               {min_len}")
print(f"  Old teacher mel_len:       {old_mel.shape[0]}")
print(f"  Mel L1 (old vs new feat):  {np.abs(old_mel[:T] - new_mel[:T]).mean():.4f}")
print(f"  A0 L1 on OLD features:     {l1_old:.4f}")
print(f"  A0 L1 on NEW features:     {l1_new:.4f}")
print(f"  L1 ratio (new/old):        {l1_new/l1_old:.2f}x")
print()
print("  Interpretation:")
if l1_new < 0.25 and abs(l1_new - l1_old) < 0.1:
    print("    PASS: No pipeline drift. Old and new features give similar A0 L1.")
    print("    Unseen failure is NOT caused by feature pipeline differences.")
elif l1_new > 0.4:
    print("    FAIL: Pipeline drift detected! Regenerated training poem also has high L1.")
    print("    Unseen failure may be partially or fully caused by pipeline differences.")
else:
    print("    MARGINAL: Some difference detected. Investigate further.")
