"""Trace the full audio pipeline to find where the timescale mismatch comes from.

Pipeline:
1. PaddleSpeech FS2 produces mel at 80fps (hop=300, sr=24000)
2. HiFiGAN vocodes mel to audio at 24kHz
3. Audio re-extracted to mel at hop=300, sr=24000 -> 80fps
4. Audio resampled to 16kHz for MFA
5. MFA aligns at 16kHz -> TextGrid

Hypothesis: MFA TextGrid timestamps are at the 16kHz audio's native
sample rate, but build_durations converts them using FRAME_RATE=80
which is based on the 24kHz mel. If MFA's internal hop/frame timing
differs from our mel hop, the timescales won't match.

Actually wait - TextGrid timestamps are in seconds, not frames.
The conversion is: frames = round(seconds * FRAME_RATE).
So if TextGrid says xmax=11.92s, that's round(11.92*80) = 954 frames.
But the actual mel has 687 frames = 8.59s.

So the question is: WHY is the TextGrid 11.92s when the audio is 8.59s?
"""
import sys, json, numpy as np, soundfile as sf, librosa
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
pid = "poem_0001"

# 1. Check the 24kHz teacher audio (paddle_distill_wav)
wav24k_path = ROOT / "output" / "paddle_distill_wav" / f"{pid}.wav"
if wav24k_path.exists():
    audio24k, sr24k = sf.read(str(wav24k_path))
    print(f"24kHz teacher wav: {len(audio24k)} samples = {len(audio24k)/sr24k:.3f}s at {sr24k}Hz")
    print(f"  Expected mel frames: {len(audio24k)/300:.0f}")
else:
    print(f"24kHz wav not found: {wav24k_path}")

# 2. Check the 16kHz MFA corpus wav
wav16k_path = ROOT / "data" / "paddle_mfa_corpus" / f"{pid}.wav"
if wav16k_path.exists():
    audio16k, sr16k = sf.read(str(wav16k_path))
    print(f"\n16kHz MFA wav: {len(audio16k)} samples = {len(audio16k)/sr16k:.3f}s at {sr16k}Hz")
else:
    print(f"16kHz wav not found: {wav16k_path}")

# 3. Check the actual mel in npz
npz = np.load(str(ROOT / "data" / "paddle_distill_features" / f"{pid}.npz"))
mel = npz["mel"]
print(f"\nMel: {mel.shape[0]} frames = {mel.shape[0]/80:.3f}s at 80fps")

# 4. Check TextGrid
tg_path = ROOT / "data" / "mfa_aligned" / f"{pid}.TextGrid"
content = tg_path.read_text(encoding="utf-8")
# Find xmax of the last interval
lines = content.split("\n")
last_xmax = None
for i, line in enumerate(lines):
    if "xmax =" in line:
        val = line.split("=")[1].strip()
        try:
            last_xmax = float(val)
        except ValueError:
            pass
print(f"\nTextGrid last xmax: {last_xmax:.3f}s")
print(f"TextGrid frames at 80fps: {round(last_xmax * 80)}")

# 5. Key question: is the 16kHz wav LONGER than the 24kHz wav?
if wav24k_path.exists() and wav16k_path.exists():
    dur24 = len(audio24k) / sr24k
    dur16 = len(audio16k) / sr16k
    print(f"\nDuration comparison:")
    print(f"  24kHz wav: {dur24:.3f}s")
    print(f"  16kHz wav: {dur16:.3f}s")
    print(f"  Ratio: {dur16/dur24:.4f}")
    print(f"  TextGrid / 16kHz: {last_xmax/dur16:.4f}")
    print(f"  Mel / 24kHz: {(mel.shape[0]/80)/dur24:.4f}")

# 6. Check extract_mel function to see how mel is extracted
from generate_paddlespeech_distillation_data import extract_mel, SR
print(f"\nSR constant: {SR}")
# Re-extract mel from 24kHz audio
if wav24k_path.exists():
    mel_reextract = extract_mel(audio24k, SR)
    print(f"Re-extracted mel: {mel_reextract.shape[0]} frames")
    print(f"Match with stored mel: {mel_reextract.shape[0] == mel.shape[0]}")
