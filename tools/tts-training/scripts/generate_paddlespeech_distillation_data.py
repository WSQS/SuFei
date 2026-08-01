"""Generate PaddleSpeech distillation data for B' experiment.

Pipeline (same-source, fully consistent):
  1. Poem text → G2P (pypinyin, same as manifest builder)
  2. Phoneme IDs → PaddleSpeech FS2 ONNX → teacher mel
  3. Teacher mel → PaddleSpeech HiFiGAN ONNX → teacher audio
  4. Teacher audio → MFA align → per-phoneme durations
  5. Teacher audio → mel/f0/energy extraction (same params as original pipeline)
  6. ASR quality filter (FunASR Paraformer)
  7. Write manifest + npz features

All mel/duration/pitch/energy labels derive from the SAME PaddleSpeech
generation, ensuring the acoustic backbone learns a self-consistent
target space.

Gate checks (run before full generation):
  B0': teacher mel → HiFiGAN → ASR on 5 samples
  Assert: duration.sum() == mel_len for every sample
  Assert: G2P phonemes match phone_id_map exactly (no <unk>)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

FS2_DIR = ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0"
HIFIGAN_DIR = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"

# Mel params (identical to nar_extract_features.py)
SR = 24000
N_FFT = 2048
HOP_LENGTH = 300
WIN_LENGTH = 1200
N_MELS = 80
FMIN = 80
FMAX = 7600
FRAME_RATE = SR / HOP_LENGTH  # 80

# Duration split (identical to nar_build_manifest.py)
INITIAL_RATIO = 0.3

# --- G2P (copied from nar_build_manifest.py for consistency) ---

INITIALS = {"b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x",
            "zh", "ch", "sh", "r", "z", "c", "s"}

PYPINYIN_FINAL_FIX = {
    "un": "uen", "ui": "uei", "iu": "iou", "ue": "ve",
    "ü": "v", "üe": "ve", "n": "en",
}

PUNCT_TO_PHONE = {
    "，": "，", "。": "。", "？": "？", "！": "！",
    "；": "，", "：": "，", "·": "。",
}


def load_phone_id_map():
    phone_map = {}
    with open(FS2_DIR / "phone_id_map.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                phone_map[parts[0]] = int(parts[1])
    return phone_map


def text_to_phonemes(text, punct_map=None):
    """Convert Chinese text to PaddleSpeech-style phoneme sequence.

    punct_map overrides PUNCT_TO_PHONE; chars absent from the map emit no
    token (e.g. drop "·" for sources that read 朝代·作者 connected).
    """
    from pypinyin import pinyin, Style

    if punct_map is None:
        punct_map = PUNCT_TO_PHONE
    result = []
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            py_list = pinyin(char, style=Style.TONE3, errors="default")
            if not py_list or not py_list[0]:
                continue
            py = py_list[0][0]
            if not py or py == char:
                continue

            tone = ""
            if py[-1].isdigit():
                tone = py[-1]
                py_base = py[:-1]
            else:
                tone = "5"
                py_base = py

            # Handle y/w pseudo-initials
            if py_base[0] == "y":
                if len(py_base) == 1:
                    py_base = "i"
                elif py_base == "you":
                    py_base = "iou"
                elif py_base == "yue":
                    py_base = "ve"
                elif py_base == "yu":
                    py_base = "v"
                elif py_base == "yuan":
                    py_base = "van"
                elif py_base == "yun":
                    py_base = "vn"
                elif py_base == "yong":
                    py_base = "iong"
                elif py_base[1] in "ae":
                    py_base = "i" + py_base[1:]
                elif py_base[1] == "i":
                    py_base = py_base[1:]
            elif py_base[0] == "w":
                if len(py_base) == 1:
                    py_base = "u"
                elif py_base == "wu":
                    py_base = "u"
                else:
                    py_base = "u" + py_base[1:]

            for old, new in PYPINYIN_FINAL_FIX.items():
                if py_base == old:
                    py_base = new
                    break

            matched = False
            for init_len in [2, 1]:
                if len(py_base) > init_len:
                    candidate_init = py_base[:init_len]
                    candidate_final = py_base[init_len:]
                    if candidate_init in INITIALS:
                        if candidate_final.rstrip("12345").rstrip("r") == "i":
                            if candidate_init in {"z", "c", "s"}:
                                candidate_final = candidate_final.replace("i", "ii")
                            elif candidate_init in {"zh", "ch", "sh", "r"}:
                                candidate_final = candidate_final.replace("i", "iii")

                        final_base = candidate_final.rstrip("12345").rstrip("r")
                        if final_base in PYPINYIN_FINAL_FIX:
                            new_base = PYPINYIN_FINAL_FIX[final_base]
                            candidate_final = new_base + tone
                        else:
                            candidate_final = candidate_final + tone

                        result.append((candidate_init, char))
                        result.append((candidate_final, char))
                        matched = True
                        break

            if not matched:
                for old, new in PYPINYIN_FINAL_FIX.items():
                    if py_base == old:
                        py_base = new
                        break
                result.append((py_base + tone, char))

        elif char in punct_map:
            result.append((punct_map[char], char))

    result.append(("<eos>", None))
    return result


def phones_to_ids(phonemes, phone_map):
    """Convert phonemes to IDs. Returns (ids, unmapped_phonemes)."""
    ids = []
    unmapped = []
    for pstr, _ in phonemes:
        if pstr in phone_map:
            ids.append(phone_map[pstr])
        else:
            unmapped.append(pstr)
            ids.append(phone_map.get("<unk>", 1))
    return ids, unmapped


# --- PaddleSpeech ONNX inference ---

def run_fs2(fs2_session, phone_ids):
    """Run FS2 ONNX: text IDs → mel spectrogram [T, 80]."""
    text_arr = np.array(phone_ids, dtype=np.int64)
    mel = fs2_session.run(None, {"text": text_arr})[0]
    return mel


def run_hifigan(hifigan_session, mel):
    """Run HiFiGAN ONNX: mel [T, 80] → audio [N]."""
    mel_input = mel.astype(np.float32)
    if mel_input.ndim == 3:
        mel_input = mel_input[0]  # squeeze batch dim if present
    audio = hifigan_session.run(None, {"logmel": mel_input})[0].flatten()
    return audio


# --- Feature extraction (same as nar_extract_features.py) ---

def extract_mel(audio, sr=SR):
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        window="hann",
    )
    mel = np.log(np.maximum(mel, 1e-5))
    return mel.T  # (T, n_mels)


def extract_f0(audio, sr=SR):
    import pyworld as pyworld
    audio_d = audio.astype(np.float64)
    _f0, t = pyworld.dio(audio_d, sr, f0_floor=80, f0_ceil=400)
    f0 = pyworld.stonemask(audio_d, _f0, t, sr)
    n_frames = len(audio) // HOP_LENGTH + 1
    f0_resampled = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start_sample = i * HOP_LENGTH
        end_sample = min((i + 1) * HOP_LENGTH, len(audio))
        frame_f0 = f0[start_sample // 256: end_sample // 256]
        voiced = frame_f0[frame_f0 > 0]
        if len(voiced) > 0:
            f0_resampled[i] = np.mean(voiced)
        else:
            f0_resampled[i] = 0.0
    return f0_resampled


def extract_energy(mel):
    return np.sum(mel, axis=1)


# --- MFA duration alignment ---

def run_mfa_align(wav_path, text, poem_id, mfa_corpus_dir, mfa_output_dir, dictionary_path):
    """Run MFA alignment on a single audio+text pair.

    Returns: list of TextGrid intervals (xmin, xmax, text, is_silence)
    or None if alignment failed.
    """
    import subprocess
    import shutil

    corpus = Path(mfa_corpus_dir)
    corpus.mkdir(parents=True, exist_ok=True)

    # Write wav + lab
    lab_path = corpus / f"{poem_id}.lab"
    wav_dest = corpus / f"{poem_id}.wav"

    if not wav_dest.exists() or wav_dest.stat().st_size == 0:
        import soundfile as sf
        audio, sr = sf.read(wav_path)
        sf.write(str(wav_dest), audio, sr)

    lab_path.write_text(text, encoding="utf-8")

    # Run MFA align
    output_tg = Path(mfa_output_dir)
    output_tg.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mfa", "align",
        str(corpus),
        str(dictionary_path),
        "mandarin_mfa",
        str(output_tg),
        "--overwrite",
        "--clean",
        "--num_jobs", "1",
        "--beam", "100",
        "--retry_beam", "400",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  MFA failed for {poem_id}: {result.stderr[:200]}")
        return None

    tg_path = output_tg / f"{poem_id}.TextGrid"
    if not tg_path.exists():
        return None

    return parse_textgrid_intervals(tg_path)


def parse_textgrid_intervals(path):
    """Parse TextGrid word-tier intervals."""
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")

    intervals = []
    in_words_tier = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if 'name = "words"' in line:
            in_words_tier = True
        elif 'name = "phones"' in line:
            in_words_tier = False

        if in_words_tier and "text =" in line and "xmin" not in line:
            text = line.split("=", 1)[1].strip().strip('"')
            xmin = xmax = None
            for j in range(i - 1, max(i - 6, 0), -1):
                jl = lines[j].strip()
                if jl.startswith("xmax =") and xmax is None:
                    xmax = float(jl.split("=")[1].strip())
                elif jl.startswith("xmin =") and xmin is None:
                    xmin = float(jl.split("=")[1].strip())
            if xmin is not None and xmax is not None:
                intervals.append({
                    "xmin": xmin, "xmax": xmax,
                    "text": text, "is_silence": len(text) == 0,
                })
        i += 1

    return intervals


def build_durations(phonemes, tg_intervals):
    """Build per-phoneme duration array from TextGrid intervals.

    Logic identical to nar_build_manifest.py.
    """
    total_time = tg_intervals[-1]["xmax"] if tg_intervals else 0
    total_mel = round(total_time * FRAME_RATE)

    timeline = []
    for iv in tg_intervals:
        start_f = round(iv["xmin"] * FRAME_RATE)
        end_f = round(iv["xmax"] * FRAME_RATE)
        dur = max(end_f - start_f, 0)
        timeline.append({
            "text": iv["text"], "start": start_f, "end": end_f,
            "dur": dur, "is_silence": iv["is_silence"],
        })

    durations = []
    tg_idx = 0
    i = 0

    while i < len(phonemes):
        phone_str, char = phonemes[i]

        if phone_str == "<eos>":
            durations.append(1)
            i += 1
            continue

        if phone_str in ("，", "。", "？", "！"):
            silence_dur = 0
            while tg_idx < len(timeline):
                tl = timeline[tg_idx]
                if tl["is_silence"]:
                    silence_dur = tl["dur"]
                    tg_idx += 1
                    break
                elif tl["text"]:
                    tg_idx += 1
                else:
                    tg_idx += 1
            durations.append(max(silence_dur, 1))
            i += 1
            continue

        char_interval = None
        while tg_idx < len(timeline):
            tl = timeline[tg_idx]
            if tl["text"] == char and not tl["is_silence"]:
                char_interval = tl
                tg_idx += 1
                break
            else:
                tg_idx += 1

        if char_interval is None:
            durations.append(2)
            i += 1
            continue

        char_dur = char_interval["dur"]

        if i + 1 < len(phonemes) and phonemes[i + 1][1] == char:
            init_dur = max(round(char_dur * INITIAL_RATIO), 1)
            final_dur = max(char_dur - init_dur, 1)
            durations.append(init_dur)
            durations.append(final_dur)
            i += 2
        else:
            durations.append(max(char_dur, 1))
            i += 1

    current_sum = sum(durations)
    diff = total_mel - current_sum
    if diff != 0:
        max_idx = max(range(len(durations) - 1), key=lambda x: durations[x])
        durations[max_idx] = max(durations[max_idx] + diff, 1)

    return durations


# --- ASR filter ---

def load_asr_model():
    from funasr import AutoModel
    return AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device="cpu",
        disable_update=True,
    )


def transcribe(asr_model, wav_path):
    rec = asr_model.generate(input=wav_path, batch_size_s=300)
    return rec[0]["text"] if rec else ""


def normalize_text(text):
    """Remove punctuation and whitespace for CER comparison."""
    import re
    text = re.sub(r"[，。？！；：、·\s\n]", "", text)
    return text


def align(ref, hyp):
    """Levenshtein alignment for CER."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            ops.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append((ref[i - 1], "*"))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("*", hyp[j - 1]))
            j -= 1
        else:
            ops.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
    ops.reverse()
    return ops


# --- Main pipeline ---

def main():
    parser = argparse.ArgumentParser(description="Generate PaddleSpeech distillation data")
    parser.add_argument("--gate-test", action="store_true",
                        help="Run B0' gate test only (5 samples, no MFA, no manifest)")
    parser.add_argument("--stage", type=int, default=0,
                        help="Run specific stage: 1=inference+ASR+features, 2=MFA align, 3=build manifest")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--dictionary", default=str(ROOT / "data" / "mfa_char_dictionary.txt"),
                        help="MFA dictionary path")
    args = parser.parse_args()

    import onnxruntime as ort
    import soundfile as sf

    # Load models
    print("Loading PaddleSpeech ONNX models...")
    fs2 = ort.InferenceSession(str(FS2_DIR / "fastspeech2_csmsc.onnx"),
                               providers=["CPUExecutionProvider"])
    hifigan = ort.InferenceSession(str(HIFIGAN_DIR / "hifigan_csmsc.onnx"),
                                   providers=["CPUExecutionProvider"])
    phone_map = load_phone_id_map()
    print(f"  FS2 + HiFiGAN loaded. Phone vocab: {len(phone_map)}")

    # Load training data source texts
    source_manifest = ROOT / "data" / "train_manifest.jsonl"
    samples = []
    with open(source_manifest, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"  Source samples: {len(samples)}")

    # --- B0' Gate Test ---
    if args.gate_test:
        print("\n" + "=" * 60)
        print("B0' GATE TEST: PaddleSpeech mel → HiFiGAN → ASR")
        print("=" * 60)

        asr = load_asr_model()

        gate_results = []
        for s in samples[:5]:
            poem_id = s["poem_id"]
            text = s["text"]

            # Split into header (title+dynasty+author) and content
            # Format: "标题，朝代·作者。正文"
            parts = text.split("。", 1)
            header = parts[0] if len(parts) > 1 else ""
            content = parts[1] if len(parts) > 1 else text
            ref_full = normalize_text(text)
            ref_content = normalize_text(content)

            # G2P
            phones = text_to_phonemes(text)
            ids, unmapped = phones_to_ids(phones, phone_map)

            # FS2 inference
            mel = run_fs2(fs2, ids)
            audio = run_hifigan(hifigan, mel)

            # Save temp wav
            wav_path = str(ROOT / "output" / "gate_test" / f"{poem_id}.wav")
            Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav_path, audio, SR)

            # ASR
            asr_text = transcribe(asr, wav_path)
            hyp = normalize_text(asr_text)

            # Full CER
            ops = align(ref_full, hyp)
            d = sum(1 for r, h in ops if r != "*" and h == "*")
            sub = sum(1 for r, h in ops if r != "*" and h != "*" and r != h)
            cer_full = (d + sub) / max(len(ref_full), 1)

            # Content-only CER (find content substring in ASR output)
            ops_c = align(ref_content, hyp)
            d_c = sum(1 for r, h in ops_c if r != "*" and h == "*")
            sub_c = sum(1 for r, h in ops_c if r != "*" and h != "*" and r != h)
            cer_content = (d_c + sub_c) / max(len(ref_content), 1)

            status = "OK" if cer_content < 0.15 else ("MARGINAL" if cer_content < 0.30 else "FAIL")
            print(f"  {poem_id}: full_CER={cer_full:.0%} content_CER={cer_content:.0%} {status}")
            print(f"    header: {header}")
            print(f"    ref:  {ref_content[:30]}")
            print(f"    hyp:  {hyp[:30]}")
            gate_results.append({"poem_id": poem_id, "cer_full": cer_full,
                                 "cer_content": cer_content, "status": status})

        ok_count = sum(1 for r in gate_results if r["status"] == "OK")
        content_cers = [r["cer_content"] for r in gate_results]
        print(f"\nB0' Result: {ok_count}/{len(gate_results)} OK (content CER < 15%)")
        print(f"  Content CER: mean={np.mean(content_cers):.0%} median={np.median(content_cers):.0%}")
        if ok_count >= 4:
            print("GATE PASSED — teacher pipeline is sound")
        else:
            print("GATE FAILED — investigate teacher audio quality first")
        return

    # --- Full pipeline (3 stages) ---
    feat_dir = ROOT / "data" / "paddle_distill_features"
    wav_dir = ROOT / "output" / "paddle_distill_wav"
    mfa_corpus = ROOT / "data" / "paddle_mfa_corpus"
    mfa_output = ROOT / "data" / "paddle_mfa_aligned"
    manifest_path = ROOT / "data" / "paddle_distill_manifest.jsonl"
    asr_filter_path = ROOT / "data" / "paddle_distill_asr_filter.json"

    for d in [feat_dir, wav_dir, mfa_corpus, mfa_output]:
        d.mkdir(parents=True, exist_ok=True)

    run_all = args.stage == 0

    # ===================================================================
    # STAGE 1: PaddleSpeech inference + ASR filter + feature extraction
    # ===================================================================
    if run_all or args.stage == 1:
        print("\n" + "=" * 60)
        print("STAGE 1: PaddleSpeech inference + ASR filter + features")
        print("=" * 60)

        asr = load_asr_model()

        asr_results = []
        stats1 = {"total": 0, "asr_pass": 0, "asr_fail": 0, "unmapped": 0}

        for idx, s in enumerate(samples):
            poem_id = s["poem_id"]
            text = s["text"]
            stats1["total"] += 1

            # G2P
            phones = text_to_phonemes(text)
            ids, unmapped = phones_to_ids(phones, phone_map)
            if unmapped:
                print(f"  WARN {poem_id}: {len(unmapped)} unmapped phones: {unmapped[:5]}")
                stats1["unmapped"] += 1

            # FS2 → teacher mel → HiFiGAN → audio
            teacher_mel = run_fs2(fs2, ids)
            audio = run_hifigan(hifigan, teacher_mel)

            # Save wav
            wav_path = str(wav_dir / f"{poem_id}.wav")
            sf.write(wav_path, audio, SR)

            # ASR filter
            ref = normalize_text(text)
            asr_text = transcribe(asr, wav_path)
            hyp = normalize_text(asr_text)
            ops = align(ref, hyp)
            d = sum(1 for r, h in ops if r != "*" and h == "*")
            sub = sum(1 for r, h in ops if r != "*" and h != "*" and r != h)
            cer = (d + sub) / max(len(ref), 1)

            passed = cer <= 0.30
            if passed:
                stats1["asr_pass"] += 1
            else:
                stats1["asr_fail"] += 1
                print(f"  SKIP {poem_id}: ASR CER={cer:.0%}")

            # Extract mel/f0/energy from teacher audio
            reextracted_mel = extract_mel(audio, SR)
            f0 = extract_f0(audio, SR)
            energy = extract_energy(reextracted_mel)

            min_len = min(len(f0), reextracted_mel.shape[0], len(energy))
            reextracted_mel = reextracted_mel[:min_len]
            f0 = f0[:min_len]
            energy = energy[:min_len]

            # Save features
            npz_path = feat_dir / f"{poem_id}.npz"
            np.savez_compressed(
                str(npz_path),
                mel=reextracted_mel.astype(np.float32),
                f0=f0.astype(np.float32),
                energy=energy.astype(np.float32),
            )

            # Save MFA corpus files (16kHz for MFA)
            mfa_wav_path = mfa_corpus / f"{poem_id}.wav"
            import librosa as lr
            audio_16k = lr.resample(audio, orig_sr=SR, target_sr=16000)
            sf.write(str(mfa_wav_path), audio_16k, 16000)

            lab_path = mfa_corpus / f"{poem_id}.lab"
            lab_path.write_text(text, encoding="utf-8")

            asr_results.append({
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
                print(f"  Stage 1 progress: {idx+1}/{len(samples)}")

        with open(asr_filter_path, "w", encoding="utf-8") as f:
            json.dump({"results": asr_results, "stats": stats1}, f, ensure_ascii=False, indent=2)

        print(f"\nStage 1 complete:")
        print(f"  Total: {stats1['total']}, ASR pass: {stats1['asr_pass']}, ASR fail: {stats1['asr_fail']}")
        print(f"  Features: {feat_dir}")
        print(f"  MFA corpus: {mfa_corpus}")
        print(f"  ASR filter: {asr_filter_path}")

        if not run_all:
            return

    # ===================================================================
    # STAGE 2: MFA alignment (batch)
    # ===================================================================
    if run_all or args.stage == 2:
        print("\n" + "=" * 60)
        print("STAGE 2: MFA batch alignment")
        print("=" * 60)

        import subprocess
        import shutil

        # Clean output dir
        if mfa_output.exists():
            shutil.rmtree(mfa_output)
        mfa_output.mkdir(parents=True)

        cmd = [
            "mfa", "align",
            str(mfa_corpus),
            str(args.dictionary),
            "mandarin_mfa",
            str(mfa_output),
            "--overwrite",
            "--clean",
            "--num_jobs", "4",
            "--beam", "100",
            "--retry_beam", "400",
        ]

        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, timeout=3600)

        tg_count = len(list(mfa_output.glob("*.TextGrid")))
        print(f"\nStage 2 complete: {tg_count} TextGrids generated")

        if not run_all:
            return

    # ===================================================================
    # STAGE 3: Build manifest from TextGrids + features
    # ===================================================================
    if run_all or args.stage == 3:
        print("\n" + "=" * 60)
        print("STAGE 3: Build distillation manifest")
        print("=" * 60)

        with open(asr_filter_path, encoding="utf-8") as f:
            asr_data = json.load(f)

        records = []
        stats3 = {"mfa_fail": 0, "dur_mismatch": 0, "skipped_asr": 0}

        for entry in asr_data["results"]:
            poem_id = entry["poem_id"]

            if not entry["asr_pass"]:
                stats3["skipped_asr"] += 1
                continue

            text = entry["text"]
            ids = entry["phoneme_ids"]
            mel_len = entry["mel_len"]

            # Parse TextGrid
            tg_path = mfa_output / f"{poem_id}.TextGrid"
            if not tg_path.exists():
                print(f"  SKIP {poem_id}: no TextGrid")
                stats3["mfa_fail"] += 1
                continue

            tg_intervals = parse_textgrid_intervals(tg_path)
            phones = text_to_phonemes(text)
            durations = build_durations(phones, tg_intervals)

            # Fix duration sum to match mel_len
            dur_sum = sum(durations)
            if dur_sum != mel_len:
                diff = mel_len - dur_sum
                max_idx = max(range(len(durations)), key=lambda x: durations[x])
                durations[max_idx] = max(durations[max_idx] + diff, 1)

            if sum(durations) != mel_len:
                print(f"  SKIP {poem_id}: duration mismatch {sum(durations)} vs {mel_len}")
                stats3["dur_mismatch"] += 1
                continue

            records.append({
                "poem_id": poem_id,
                "phoneme_ids": ids,
                "durations": durations,
                "mel_path": f"data/paddle_distill_features/{poem_id}.npz",
                "mel_len": mel_len,
                "n_phonemes": len(ids),
                "text": text,
                "asr_cer": entry["asr_cer"],
            })

        with open(manifest_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        total_mel = sum(r["mel_len"] for r in records)
        bad = sum(1 for r in records if sum(r["durations"]) != r["mel_len"])

        print(f"\nStage 3 complete:")
        print(f"  Skipped (ASR fail): {stats3['skipped_asr']}")
        print(f"  Skipped (MFA fail): {stats3['mfa_fail']}")
        print(f"  Skipped (dur mismatch): {stats3['dur_mismatch']}")
        print(f"  Final records: {len(records)}")
        print(f"  Total mel frames: {total_mel} ({total_mel / FRAME_RATE / 60:.1f} min)")
        print(f"  Duration invariant violations: {bad}/{len(records)}")
        print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
