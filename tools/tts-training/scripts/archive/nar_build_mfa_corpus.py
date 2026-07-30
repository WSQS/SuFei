"""Build MFA alignment corpus from clean teacher data.

Structure:
  mfa_corpus/
    {poem_id}.wav   (16kHz mono, resampled from 24kHz)
    {poem_id}.txt   (full text including title+author+content)

MFA expects matching filenames for .wav and .lab/.txt.
"""
import json
import re
from pathlib import Path

import torchaudio

ROOT = Path(__file__).resolve().parents[1]
CLEAN_JSON = ROOT / "data" / "clean_data_list.json"
CORPUS_DIR = ROOT / "data" / "mfa_corpus"
TARGET_SR = 16000


def main():
    with open(CLEAN_JSON, encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for s in samples:
        audio_path = Path(s["audio"])
        if not audio_path.exists():
            print(f"  SKIP: {audio_path.name} not found")
            continue

        # Use index as poem_id for MFA (avoid hash collisions)
        poem_id = f"poem_{s['index']:04d}"

        # Resample to 16kHz mono
        wav, sr = torchaudio.load(str(audio_path))
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        out_wav = CORPUS_DIR / f"{poem_id}.wav"
        torchaudio.save(str(out_wav), wav, TARGET_SR)

        # Write text file (full text with title+author)
        out_txt = CORPUS_DIR / f"{poem_id}.lab"
        # MFA .lab format: plain text, one utterance per file
        text = s["full_text"]
        # Normalize punctuation to basic forms MFA handles
        # Keep commas and periods as they may help MFA with pauses
        out_txt.write_text(text, encoding="utf-8")

        count += 1

    print(f"Built corpus: {count} files in {CORPUS_DIR}")

    # Verify
    wavs = list(CORPUS_DIR.glob("*.wav"))
    labs = list(CORPUS_DIR.glob("*.lab"))
    print(f"  WAVs: {len(wavs)}, LABs: {len(labs)}")


if __name__ == "__main__":
    main()
