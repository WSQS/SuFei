"""Check if the TextGrid timestamps make sense relative to actual audio content.

The audio is 8.575s but TextGrid says 11.92s. Let's check:
1. Are the word interval timestamps within the audio duration?
2. Is the speaking rate reasonable?
3. What happens after the last word interval?
"""
import soundfile as sf, numpy as np
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
pid = "poem_0001"

# Read audio
audio, sr = sf.read(str(ROOT / "data" / "paddle_mfa_corpus" / f"{pid}.wav"))
audio_dur = len(audio) / sr
print(f"Audio: {audio_dur:.3f}s ({len(audio)} samples @ {sr}Hz)")

# Parse ALL intervals from TextGrid
content = (ROOT / "data" / "mfa_aligned" / f"{pid}.TextGrid").read_text(encoding="utf-8")
lines = content.split("\n")

words_intervals = []
phones_intervals = []
current_tier = None

i = 0
while i < len(lines):
    line = lines[i].strip()
    if 'name = "words"' in line:
        current_tier = "words"
    elif 'name = "phones"' in line:
        current_tier = "phones"
    
    if line.startswith("xmin =") and current_tier:
        xmin = float(line.split("=")[1].strip())
        # Look for xmax and text in nearby lines
        xmax = None
        text = None
        for j in range(i+1, min(i+6, len(lines))):
            jl = lines[j].strip()
            if jl.startswith("xmax =") and xmax is None:
                xmax = float(jl.split("=")[1].strip())
            elif jl.startswith("text =") and text is None:
                text = jl.split("=", 1)[1].strip().strip('"')
        if xmax is not None:
            iv = {"xmin": xmin, "xmax": xmax, "text": text or "",
                  "dur": xmax - xmin, "tier": current_tier}
            if current_tier == "words":
                words_intervals.append(iv)
            else:
                phones_intervals.append(iv)
    i += 1

print(f"\nWords tier: {len(words_intervals)} intervals")
print(f"Phones tier: {len(phones_intervals)} intervals")

# Show last 5 word intervals
print(f"\nLast 5 word intervals:")
for iv in words_intervals[-5:]:
    print(f"  {iv['xmin']:.3f} - {iv['xmax']:.3f} ({iv['dur']:.3f}s) [{iv['text']}]")

# Check: does the last word interval end at audio_dur?
last_word_end = max(iv["xmax"] for iv in words_intervals)
print(f"\nLast word interval end: {last_word_end:.3f}s")
print(f"Audio duration: {audio_dur:.3f}s")
print(f"TextGrid xmax: 11.920s")

# Check speaking rate
speech_words = [iv for iv in words_intervals if iv["text"]]
total_speech = sum(iv["dur"] for iv in speech_words)
total_silence = sum(iv["dur"] for iv in words_intervals if not iv["text"])
print(f"\nTotal speech: {total_speech:.3f}s ({len(speech_words)} words)")
print(f"Total silence: {total_silence:.3f}s")
print(f"Speech rate: {len(speech_words)/total_speech:.1f} chars/s")

# The speaking rate for Chinese TTS should be ~4-6 chars/s
# If it's ~7+ chars/s, the timestamps are compressed
print(f"Expected ~4-6 chars/s for clear speech")

# Key: the word intervals make sense timing-wise within the audio,
# but the TextGrid xmax (11.92) extends well beyond the audio (8.575).
# MFA likely added padding to the TextGrid.

# Let's check: are all word intervals within audio duration?
beyond = [iv for iv in words_intervals if iv["xmin"] > audio_dur]
print(f"\nIntervals starting after audio end: {len(beyond)}")
if beyond:
    for iv in beyond[:5]:
        print(f"  {iv['xmin']:.3f} - {iv['xmax']:.3f} [{iv['text']}]")

# Check: maybe the intervals are correct but we need to clip to audio duration
clipped_intervals = [iv for iv in words_intervals if iv["xmax"] <= audio_dur]
print(f"\nIntervals within audio duration: {len(clipped_intervals)}/{len(words_intervals)}")
beyond2 = [iv for iv in words_intervals if iv["xmin"] < audio_dur < iv["xmax"]]
print(f"Intervals straddling audio end: {len(beyond2)}")
for iv in beyond2:
    print(f"  {iv['xmin']:.3f} - {iv['xmax']:.3f} [{iv['text']}]")

# Show the actual word timestamps mapped to mel frames
print(f"\n--- Word timestamps -> mel frames (clipped to {audio_dur:.3f}s) ---")
FRAME_RATE = 80
for iv in words_intervals[:20]:
    mel_start = round(iv["xmin"] * FRAME_RATE)
    mel_end = min(round(iv["xmax"] * FRAME_RATE), round(audio_dur * FRAME_RATE))
    if mel_end > mel_start:
        print(f"  [{iv['xmin']:.3f}-{iv['xmax']:.3f}s] -> frames [{mel_start}-{mel_end}] "
              f"(dur={mel_end-mel_start}) [{iv['text']}]")
