"""Extract Mel spectrogram, F0, and energy from teacher audio.

Output: data/nar_features/ with per-sample .npz files containing:
  - mel: (T, 80) log-mel spectrogram
  - f0: (T,) fundamental frequency
  - energy: (T,) energy per frame
  - duration: (L,) per-phoneme duration in mel frames

Uses PaddleSpeech-compatible parameters:
  sr=24000, n_fft=2048, hop=300, win=1200, n_mels=80, fmin=80, fmax=7600
"""
import json
import math
from pathlib import Path

import numpy as np
import librosa
import pyworld as pyworld
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
DURATION_JSON = ROOT / "data" / "duration_labels.json"
OUTPUT_DIR = ROOT / "data" / "nar_features"

# Mel params (matching PaddleSpeech FS2 CSMSC)
SR = 24000
N_FFT = 2048
HOP_LENGTH = 300
WIN_LENGTH = 1200
N_MELS = 80
FMIN = 80
FMAX = 7600

# F0 params
F0_MIN = 80  # Hz
F0_MAX = 400  # Hz


def extract_mel(audio, sr):
    """Extract log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        window="hann",
    )
    # Convert to log scale
    mel = np.log(np.maximum(mel, 1e-5))
    return mel.T  # (T, n_mels)


def extract_f0(audio, sr):
    """Extract F0 using pyworld."""
    # pyworld requires double
    audio_d = audio.astype(np.float64)
    _f0, t = pyworld.dio(audio_d, sr, f0_floor=F0_MIN, f0_ceil=F0_MAX)
    f0 = pyworld.stonemask(audio_d, _f0, t, sr)

    # Resample F0 to mel frame rate
    n_frames = len(audio) // HOP_LENGTH + 1
    f0_resampled = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        # Average F0 in this frame's time range
        start_sample = i * HOP_LENGTH
        end_sample = min((i + 1) * HOP_LENGTH, len(audio))
        frame_f0 = f0[start_sample // 256: end_sample // 256]  # pyworld uses 256 hop
        voiced = frame_f0[frame_f0 > 0]
        if len(voiced) > 0:
            f0_resampled[i] = np.mean(voiced)
        else:
            f0_resampled[i] = 0.0

    return f0_resampled


def extract_energy(mel):
    """Extract energy from mel spectrogram (sum of dB values per frame)."""
    return np.sum(mel, axis=1)  # (T,)


def main():
    with open(DURATION_JSON, encoding="utf-8") as f:
        dur_data = json.load(f)

    samples = dur_data["samples"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    errors = []

    for s in samples:
        poem_id = s["poem_id"]
        audio_path = s["audio_path"]
        char_durations = s["char_durations"]

        if not Path(audio_path).exists():
            errors.append(f"{poem_id}: audio not found")
            continue

        # Load audio
        wav, sr = torchaudio.load(audio_path)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        audio = wav[0].numpy()  # mono

        # Extract features
        mel = extract_mel(audio, SR)
        f0 = extract_f0(audio, SR)

        # Align F0 length to mel length
        if len(f0) < mel.shape[0]:
            f0 = np.pad(f0, (0, mel.shape[0] - len(f0)))
        elif len(f0) > mel.shape[0]:
            f0 = f0[:mel.shape[0]]

        energy = extract_energy(mel)

        # Ensure all same length
        min_len = min(len(f0), mel.shape[0], len(energy))
        mel = mel[:min_len]
        f0 = f0[:min_len]
        energy = energy[:min_len]

        # Extract per-char durations in mel frames
        durations = []
        for cd in char_durations:
            if cd["type"] == "char" and cd.get("duration_sec", 0) > 0:
                start_frame = round(cd["start"] * SR / HOP_LENGTH)
                end_frame = round(cd["end"] * SR / HOP_LENGTH)
                d = max(end_frame - start_frame, 1)
                durations.append(d)
            elif cd["type"] == "char":
                durations.append(0)  # unmatched char
            elif cd["type"] == "punct":
                durations.append(0)  # punctuation (will be handled by model)

        durations = np.array(durations, dtype=np.int32)

        # Save
        out_path = OUTPUT_DIR / f"{poem_id}.npz"
        np.savez_compressed(
            str(out_path),
            mel=mel.astype(np.float32),
            f0=f0.astype(np.float32),
            energy=energy.astype(np.float32),
            durations=durations,
        )

        success += 1
        if success % 50 == 0:
            print(f"  {success}/{len(samples)} done")

    print(f"\n{'='*60}")
    print(f"Feature extraction complete")
    print(f"  Success: {success}/{len(samples)}")
    print(f"  Errors: {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"    {e}")
    print(f"  Output: {OUTPUT_DIR}")

    # Verify one sample
    if success > 0:
        sample_path = OUTPUT_DIR / f"{samples[0]['poem_id']}.npz"
        data = np.load(str(sample_path))
        print(f"\n  Sample {samples[0]['poem_id']}:")
        print(f"    mel: {data['mel'].shape}, range=[{data['mel'].min():.2f}, {data['mel'].max():.2f}]")
        print(f"    f0: {data['f0'].shape}, voiced={np.sum(data['f0'] > 0)}/{len(data['f0'])}")
        print(f"    energy: {data['energy'].shape}")
        print(f"    durations: {data['durations'].shape}, sum={data['durations'].sum()}")


if __name__ == "__main__":
    main()
