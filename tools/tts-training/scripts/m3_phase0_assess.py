"""M3 Phase 0: ASR gate + vocoder ceiling assessment (runs on rtx).

Follows eval_v10.py conventions. For each {poem_id}_{variant}.wav:
  1. Direct CER: ASR transcribe -> cer_detail.
  2. Ceiling CER: extract log-mel with the SAME params used to build
     paddle_distill_features_v3 -> HiFiGAN ONNX -> save vocoded wav ->
     ASR -> CER.

Mel-extraction source: generate_paddlespeech_distillation_data.py
(function extract_mel + constant block SR/N_FFT/HOP_LENGTH/WIN_LENGTH/
N_MELS/FMIN/FMAX), the pipeline that produced the v2/v3 teacher mel.
"""
import argparse
import json
import sys
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

# --- Mel params (reused from generate_paddlespeech_distillation_data.py) ---
# Those are the exact constants that built paddle_distill_features_v2
# (teacher mel), which rebuild_data_v3.py copied verbatim into v3.
SR = 24000
N_FFT = 2048
HOP_LENGTH = 300
WIN_LENGTH = 1200
N_MELS = 80
FMIN = 80
FMAX = 7600

HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0'
TARGET_SR = 24000


def extract_mel(audio, sr=SR):
    """Log-mel extraction in the CALIBRATED HiFiGAN CSMSC convention.

    Calibrated by m3_mel_calibration.py (2026-07-31): magnitude spectrogram
    (power=1) + log10 + eps 1e-10 reconstructs teacher-native mel with raw
    L1=0.102 and near-identity affine fit (a=0.982, b=-0.028). The old
    librosa power=2 + ln convention was 43x worse (L1=4.44) — it caused the
    historical 31.8% copy-synthesis ceiling.
    """
    import librosa
    S = np.abs(librosa.stft(
        y=audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window='hann',
    ))
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    mel = np.maximum(np.dot(mel_basis, S), 1e-10)
    return np.log10(mel).T.astype(np.float32)  # (T, n_mels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir',
                        default='E:/sufei-training/data/m3_cosyvoice_phase0')
    parser.add_argument('--poems',
                        default='E:/sufei-training/data/m3_phase0_poems.json')
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    with open(args.poems, encoding='utf-8') as f:
        poems = json.load(f)

    print("Mel-extraction: CALIBRATED HiFiGAN convention (power=1, log10, "
          "eps 1e-10; see m3_mel_calibration.py) with params SR=%d N_FFT=%d "
          "HOP=%d WIN=%d N_MELS=%d FMIN=%d FMAX=%d" % (
              SR, N_FFT, HOP_LENGTH, WIN_LENGTH, N_MELS, FMIN, FMAX))

    print("Loading ASR + HiFiGAN ONNX...")
    asr = load_asr_model()
    voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'),
                               providers=['CPUExecutionProvider'])

    text_by_id = {p['poem_id']: p['text'] for p in poems}

    rows = []  # (pid, variant, dur_s, direct_cer, ceiling_cer)
    for wpath in sorted(wav_dir.glob('*.wav')):
        if wpath.name.endswith('_voc.wav'):
            continue
        stem = wpath.stem  # e.g. poem123_plain
        if '_' not in stem:
            continue
        pid, variant = stem.rsplit('_', 1)
        if pid not in text_by_id:
            continue
        ref_text = text_by_id[pid]

        # 1. Direct CER
        audio, sr = sf.read(str(wpath))
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
        dur_s = len(audio) / TARGET_SR
        asr_text, _ = transcribe(asr, str(wpath))
        direct_cer, _, _, _, _ = cer_detail(ref_text, asr_text)

        # 2. Ceiling CER: extract mel -> HiFiGAN -> ASR
        mel = extract_mel(audio.astype(np.float32), SR)
        voc_audio = voc.run(None, {'logmel': mel.astype(np.float32)})[0].flatten()
        voc_path = wpath.with_name(wpath.stem + '_voc.wav')
        sf.write(str(voc_path), voc_audio, SR)
        asr_text_voc, _ = transcribe(asr, str(voc_path))
        ceiling_cer, _, _, _, _ = cer_detail(ref_text, asr_text_voc)

        rows.append((pid, variant, dur_s, direct_cer, ceiling_cer))
        print("  %s_%s: dur=%.2fs direct=%.0f%% ceiling=%.0f%%" % (
            pid, variant, dur_s, direct_cer * 100, ceiling_cer * 100))

    # --- Table ---
    print("\n%-20s | %-8s | %6s | %9s | %9s" % (
        'poem_id', 'variant', 'dur_s', 'direct_CER', 'ceiling_CER'))
    print('-' * 62)
    for pid, variant, dur_s, d_cer, c_cer in rows:
        print("%-20s | %-8s | %6.2f | %8.0f%% | %8.0f%%" % (
            pid, variant, dur_s, d_cer * 100, c_cer * 100))

    # --- Per-variant means ---
    variants = sorted(set(r[1] for r in rows))
    print("\nPer-variant means:")
    for v in variants:
        v_rows = [r for r in rows if r[1] == v]
        d_mean = np.mean([r[3] for r in v_rows])
        c_mean = np.mean([r[4] for r in v_rows])
        n_full = sum(1 for r in v_rows if r[3] >= 0.99)
        print("  %-8s: n=%d  direct=%.0f%%  ceiling=%.0f%%  n(100%%)=%d" % (
            v, len(v_rows), d_mean * 100, c_mean * 100, n_full))


if __name__ == '__main__':
    main()
