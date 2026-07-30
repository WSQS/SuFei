"""Forensic analysis of greedy audio: RMS, silence ratio, frequency analysis."""
import json
import numpy as np
import torchaudio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GREEDY_DIR = ROOT / "output" / "test_audio_v4_greedy"
SAMPLE_DIR = ROOT / "output" / "test_audio_v4"
TEACHER_DIR = ROOT / "data" / "audio"

# Representative samples
SAMPLES = [
    ("wujue_01_jingyesi", "静夜思"),
    ("qilv_01_jinse", "锦瑟"),
    ("changshi_01_jiangjinjiu", "将进酒"),
]


def analyze_audio(wav_path):
    waveform, sr = torchaudio.load(str(wav_path))
    audio = waveform[0].numpy()  # mono

    # RMS
    rms = np.sqrt(np.mean(audio ** 2))

    # Peak
    peak = np.max(np.abs(audio))

    # Silence ratio (frames below threshold)
    frame_size = int(sr * 0.02)  # 20ms frames
    n_frames = len(audio) // frame_size
    silence_threshold = 0.001
    silent_frames = 0
    for i in range(n_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        frame_rms = np.sqrt(np.mean(frame ** 2))
        if frame_rms < silence_threshold:
            silent_frames += 1
    silence_ratio = silent_frames / n_frames if n_frames > 0 else 1.0

    # Spectral analysis
    fft = np.fft.rfft(audio[:sr])  # first second
    magnitude = np.abs(fft)
    spectral_centroid = np.sum(np.arange(len(magnitude)) * magnitude) / (np.sum(magnitude) + 1e-10)

    # First non-silent frame
    first_voiced = -1
    for i in range(n_frames):
        frame = audio[i * frame_size:(i + 1) * frame_size]
        frame_rms = np.sqrt(np.mean(frame ** 2))
        if frame_rms > silence_threshold:
            first_voiced = i
            break

    return {
        "sample_rate": sr,
        "duration": len(audio) / sr,
        "rms": rms,
        "peak": peak,
        "silence_ratio": silence_ratio,
        "spectral_centroid": spectral_centroid,
        "first_voiced_frame": first_voiced,
        "first_voiced_time": first_voiced * 0.02 if first_voiced >= 0 else None,
        "n_samples": len(audio),
    }


def main():
    print(f"{'Sample':<25} {'Source':<10} {'Duration':>8} {'RMS':>10} {'Peak':>10} {'Silence%':>9} {'Centroid':>9} {'1stVoiced':>10}")
    print("-" * 100)

    for pid, title in SAMPLES:
        # Greedy
        greedy_wav = GREEDY_DIR / f"{pid}_v4_greedy.wav"
        if greedy_wav.exists():
            stats = analyze_audio(greedy_wav)
            print(f"{title:<25} {'greedy':<10} {stats['duration']:8.1f} {stats['rms']:10.6f} {stats['peak']:10.6f} {stats['silence_ratio']*100:9.1f} {stats['spectral_centroid']:9.0f} {stats['first_voiced_time'] if stats['first_voiced_time'] else 'never':>10}")

        # Sampling
        sample_wav = SAMPLE_DIR / f"{pid}_v4.wav"
        if sample_wav.exists():
            stats = analyze_audio(sample_wav)
            print(f"{title:<25} {'sample':<10} {stats['duration']:8.1f} {stats['rms']:10.6f} {stats['peak']:10.6f} {stats['silence_ratio']*100:9.1f} {stats['spectral_centroid']:9.0f} {stats['first_voiced_time'] if stats['first_voiced_time'] else 'never':>10}")

        # Teacher
        # Find teacher wav by matching
        with open(ROOT / "data" / "test_set_v1.json", encoding="utf-8") as f:
            test_set = json.load(f)
        for s in test_set["samples"]:
            if s["poem_id"] == pid:
                teacher_name = s.get("teacher_wav")
                if teacher_name:
                    teacher_wav = TEACHER_DIR / teacher_name
                    if teacher_wav.exists():
                        stats = analyze_audio(teacher_wav)
                        print(f"{title:<25} {'teacher':<10} {stats['duration']:8.1f} {stats['rms']:10.6f} {stats['peak']:10.6f} {stats['silence_ratio']*100:9.1f} {stats['spectral_centroid']:9.0f} {stats['first_voiced_time'] if stats['first_voiced_time'] else 'never':>10}")
                break

        print()


if __name__ == "__main__":
    main()
