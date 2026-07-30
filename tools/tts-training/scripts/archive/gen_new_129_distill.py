"""Generate distillation data for 129 new poems.

Reads new_129_manifest.jsonl, runs the full pipeline:
  1. G2P → FS2 ONNX → HiFiGAN ONNX → audio
  2. Feature extraction (mel/f0/energy)
  3. ASR quality filter
  4. MFA alignment
  5. Build manifest entries

Outputs:
  - Features: data/paddle_distill_features/poem_03XX.npz
  - MFA corpus: data/paddle_mfa_corpus/poem_03XX.{wav,lab}
  - Manifest: data/new_129_distilled_manifest.jsonl
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from generate_paddlespeech_distillation_data import (
    text_to_phonemes,
    phones_to_ids,
    load_phone_id_map,
    run_fs2,
    run_hifigan,
    extract_mel,
    extract_f0,
    extract_energy,
    transcribe,
    normalize_text,
    align,
    load_asr_model,
    parse_textgrid_intervals,
    build_durations,
    FS2_DIR,
    HIFIGAN_DIR,
    SR,
    FRAME_RATE,
)

import onnxruntime as ort
import soundfile as sf

FEAT_DIR = os.path.join(ROOT, "data", "paddle_distill_features")
WAV_DIR = os.path.join(ROOT, "output", "paddle_distill_wav")
MFA_CORPUS = os.path.join(ROOT, "data", "paddle_mfa_corpus")
MFA_OUTPUT = os.path.join(ROOT, "data", "paddle_mfa_aligned")
MANIFEST_OUT = os.path.join(ROOT, "data", "new_129_distilled_manifest.jsonl")

DICTIONARY = os.path.join(ROOT, "data", "mfa_char_dictionary.txt")


def main():
    import subprocess
    import shutil
    import librosa as lr

    for d in [FEAT_DIR, WAV_DIR, MFA_CORPUS, MFA_OUTPUT]:
        os.makedirs(d, exist_ok=True)

    # Load new poems
    src_path = os.path.join(ROOT, "data", "new_129_manifest.jsonl")
    samples = []
    with open(src_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print("New poems to distill: %d" % len(samples))

    # Load models
    print("Loading PaddleSpeech ONNX models...")
    fs2 = ort.InferenceSession(
        str(os.path.join(FS2_DIR, "fastspeech2_csmsc.onnx")),
        providers=["CPUExecutionProvider"],
    )
    hifigan = ort.InferenceSession(
        str(os.path.join(HIFIGAN_DIR, "hifigan_csmsc.onnx")),
        providers=["CPUExecutionProvider"],
    )
    phone_map = load_phone_id_map()
    print("  Models loaded. Vocab: %d" % len(phone_map))

    # ── Stage 1: Inference + features + ASR filter ──
    print("\n" + "=" * 60)
    print("STAGE 1: Inference + features + ASR filter")
    print("=" * 60)

    asr = load_asr_model()

    stage1_results = []
    n_pass = 0
    n_fail = 0

    for idx, s in enumerate(samples):
        poem_id = s["poem_id"]
        text = s["text"]

        # G2P
        phones = text_to_phonemes(text)
        ids, unmapped = phones_to_ids(phones, phone_map)
        if unmapped:
            print("  WARN %s: %d unmapped: %s" % (poem_id, len(unmapped), unmapped[:5]))

        # FS2 → mel → HiFiGAN → audio
        teacher_mel = run_fs2(fs2, ids)
        audio = run_hifigan(hifigan, teacher_mel)

        # Save wav (24kHz)
        wav_path = os.path.join(WAV_DIR, "%s.wav" % poem_id)
        sf.write(wav_path, audio, SR)

        # ASR filter
        ref = normalize_text(text)
        asr_text, _ = transcribe(asr, wav_path)
        hyp = normalize_text(asr_text)
        ops = align(ref, hyp)
        d = sum(1 for r, h in ops if r != "*" and h == "*")
        sub = sum(1 for r, h in ops if r != "*" and h != "*" and r != h)
        cer = (d + sub) / max(len(ref), 1)

        passed = cer <= 0.30
        if passed:
            n_pass += 1
        else:
            n_fail += 1
            print("  SKIP %s: ASR CER=%.0f%%" % (poem_id, cer * 100))

        # Extract features
        reextracted_mel = extract_mel(audio, SR)
        f0 = extract_f0(audio, SR)
        energy = extract_energy(reextracted_mel)

        min_len = min(len(f0), reextracted_mel.shape[0], len(energy))
        reextracted_mel = reextracted_mel[:min_len]
        f0 = f0[:min_len]
        energy = energy[:min_len]

        # Save features
        npz_path = os.path.join(FEAT_DIR, "%s.npz" % poem_id)
        np.savez_compressed(
            str(npz_path),
            mel=reextracted_mel.astype(np.float32),
            f0=f0.astype(np.float32),
            energy=energy.astype(np.float32),
        )

        # MFA corpus (16kHz)
        mfa_wav_path = os.path.join(MFA_CORPUS, "%s.wav" % poem_id)
        audio_16k = lr.resample(audio, orig_sr=SR, target_sr=16000)
        sf.write(str(mfa_wav_path), audio_16k, 16000)

        lab_path = os.path.join(MFA_CORPUS, "%s.lab" % poem_id)
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(text)

        stage1_results.append({
            "poem_id": poem_id,
            "text": text,
            "phoneme_ids": ids,
            "n_phonemes": len(ids),
            "mel_len": min_len,
            "asr_cer": cer,
            "asr_pass": passed,
            "unmapped": unmapped,
        })

        if (idx + 1) % 20 == 0:
            print("  Stage 1: %d/%d (pass=%d, fail=%d)" % (
                idx + 1, len(samples), n_pass, n_fail))

    print("\nStage 1 done: pass=%d, fail=%d" % (n_pass, n_fail))

    # ��─ Stage 2: MFA alignment ──
    print("\n" + "=" * 60)
    print("STAGE 2: MFA batch alignment (new poems only)")
    print("=" * 60)

    # Create temp corpus dir with only new poems
    temp_corpus = os.path.join(ROOT, "data", "mfa_new129_corpus")
    temp_output = os.path.join(ROOT, "data", "mfa_new129_aligned")
    os.makedirs(temp_corpus, exist_ok=True)
    if os.path.exists(temp_output):
        shutil.rmtree(temp_output)
    os.makedirs(temp_output, exist_ok=True)

    for s in stage1_results:
        if not s["asr_pass"]:
            continue
        pid = s["poem_id"]
        src_wav = os.path.join(MFA_CORPUS, "%s.wav" % pid)
        src_lab = os.path.join(MFA_CORPUS, "%s.lab" % pid)
        if os.path.exists(src_wav):
            shutil.copy2(src_wav, temp_corpus)
        if os.path.exists(src_lab):
            shutil.copy2(src_lab, temp_corpus)

    cmd = [
        "mfa", "align",
        str(temp_corpus),
        str(DICTIONARY),
        "mandarin_mfa",
        str(temp_output),
        "--overwrite",
        "--clean",
        "--num_jobs", "4",
        "--beam", "100",
        "--retry_beam", "400",
    ]
    print("  Running MFA...")
    result = subprocess.run(cmd, capture_output=False, timeout=3600)
    tg_count = len([f for f in os.listdir(temp_output) if f.endswith(".TextGrid")])
    print("  MFA done: %d TextGrids" % tg_count)

    # ── Stage 3: Build manifest ──
    print("\n" + "=" * 60)
    print("STAGE 3: Build manifest")
    print("=" * 60)

    records = []
    for entry in stage1_results:
        if not entry["asr_pass"]:
            continue

        poem_id = entry["poem_id"]
        text = entry["text"]
        ids = entry["phoneme_ids"]
        mel_len = entry["mel_len"]

        tg_path = os.path.join(temp_output, "%s.TextGrid" % poem_id)
        if not os.path.exists(tg_path):
            print("  SKIP %s: no TextGrid" % poem_id)
            continue

        tg_intervals = parse_textgrid_intervals(tg_path)
        phones = text_to_phonemes(text)
        durations = build_durations(phones, tg_intervals)

        dur_sum = sum(durations)
        if dur_sum != mel_len:
            diff = mel_len - dur_sum
            max_idx = max(range(len(durations)), key=lambda x: durations[x])
            durations[max_idx] = max(durations[max_idx] + diff, 1)

        if sum(durations) != mel_len:
            print("  SKIP %s: dur mismatch %d vs %d" % (poem_id, sum(durations), mel_len))
            continue

        records.append({
            "poem_id": poem_id,
            "phoneme_ids": ids,
            "durations": durations,
            "mel_path": "data/paddle_distill_features/%s.npz" % poem_id,
            "mel_len": mel_len,
            "n_phonemes": len(ids),
            "text": text,
            "asr_cer": entry["asr_cer"],
        })

    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Stage 3 done: %d records" % len(records))
    print("Manifest: %s" % MANIFEST_OUT)

    # Cleanup temp dirs
    shutil.rmtree(temp_corpus, ignore_errors=True)
    shutil.rmtree(temp_output, ignore_errors=True)


if __name__ == "__main__":
    main()
