r"""
TTS content correctness evaluator.

Detects: early stop, missing lines, repeated lines, character errors.
Architecture-agnostic: any TTS that outputs WAV can be evaluated.

Usage:
    python scripts/eval_tts.py --test-set data/test_set_v1.json --audio-dir output/verify_v4 --model-version moss_v4
    python scripts/eval_tts.py --test-set data/test_set_v1.json --audio-dir data/audio --model-version teacher
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]


# === Text normalization ===

PUNCT_PATTERN = re.compile(r"[，。！？；：、,.!?;:\"'""''（）()\[\]【】《》〈〉…—\-．\.\n\r\s\u3000]")


def normalize_text(text: str) -> str:
    """Strip punctuation, whitespace, unify to simplified."""
    try:
        import opencc
        if not hasattr(normalize_text, "_converter"):
            normalize_text._converter = opencc.OpenCC("t2s")
        text = normalize_text._converter.convert(text)
    except ImportError:
        pass
    text = PUNCT_PATTERN.sub("", text)
    return text


def extract_content(text: str) -> str:
    """Extract poem content from full text (remove title/author prefix)."""
    parts = text.split("。", 1)
    if len(parts) > 1:
        return parts[1]
    return text


def strip_prefix(asr_text: str, prefix: str) -> str:
    """Strip title/author prefix from ASR output if present at the start."""
    if not prefix:
        return asr_text
    if asr_text.startswith(prefix):
        return asr_text[len(prefix):]
    for i in range(min(len(prefix), len(asr_text)), 0, -1):
        if prefix[:i] == asr_text[:i] and i >= 3:
            return asr_text[i:]
    return asr_text


# === Levenshtein alignment with operation tracking ===

def align(ref: str, hyp: str) -> List[Tuple[str, str]]:
    """Return list of (ref_char, hyp_char) pairs. ref_char='*' for insertion, hyp_char='*' for deletion."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Backtrace
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            ops.append((ref[i-1], hyp[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i-1][j] <= dp[i][j-1]):
            if dp[i-1][j] < dp[i-1][j-1] or j == 0:
                ops.append((ref[i-1], "*"))
                i -= 1
            else:
                ops.append(("*", hyp[j-1]))
                j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] < dp[i-1][j-1]):
            ops.append(("*", hyp[j-1]))
            j -= 1
        else:
            ops.append((ref[i-1], hyp[j-1]))
            i -= 1
            j -= 1

    ops.reverse()
    return ops


def compute_cer(ref: str, hyp: str) -> Dict:
    """Compute CER with D/S/I breakdown and trailing deletion."""
    ops = align(ref, hyp)

    n = len(ref)
    s = sum(1 for r, h in ops if r != "*" and h != "*" and r != h)
    d = sum(1 for r, h in ops if r != "*" and h == "*")
    ins = sum(1 for r, h in ops if r == "*" and h != "*")

    cer = (s + d + ins) / n if n > 0 else 0.0
    sub_rate = s / n if n > 0 else 0.0
    del_rate = d / n if n > 0 else 0.0
    ins_rate = ins / n if n > 0 else 0.0

    # Trailing deletion: count consecutive deletions from the end of ref
    trailing_del = 0
    for r, h in reversed(ops):
        if r != "*" and h == "*":
            trailing_del += 1
        elif r != "*" and h != "*":
            break
    trailing_del_ratio = trailing_del / n if n > 0 else 0.0

    return {
        "cer": round(cer, 4),
        "substitution_rate": round(sub_rate, 4),
        "deletion_rate": round(del_rate, 4),
        "insertion_rate": round(ins_rate, 4),
        "trailing_deletion": trailing_del,
        "trailing_deletion_ratio": round(trailing_del_ratio, 4),
        "ref_len": n,
        "S": s,
        "D": d,
        "I": ins,
    }


# === Line-level coverage ===

def compute_line_coverage(ref_lines: List[str], hyp: str) -> List[float]:
    """For each line, compute fraction of characters found in hyp (in order)."""
    coverages = []
    search_start = 0
    for line in ref_lines:
        line_norm = normalize_text(line)
        if not line_norm:
            coverages.append(1.0)
            continue

        # Greedy sequential match: find chars of this line in hyp starting from search_start
        found = 0
        pos = search_start
        for ch in line_norm:
            idx = hyp.find(ch, pos)
            if idx >= 0:
                found += 1
                pos = idx + 1
        coverage = found / len(line_norm) if line_norm else 1.0
        coverages.append(round(coverage, 4))

    return coverages


# === Repeat detection ===

def detect_repeats(ref: str, hyp: str) -> Dict:
    """Detect repeated spans in hyp that exceed what's in ref."""
    repeats = []

    # Check for repeated substrings of length >= 4
    for length in range(4, min(len(hyp) // 2 + 1, 20)):
        for start in range(len(hyp) - 2 * length + 1):
            span = hyp[start:start + length]
            next_span = hyp[start + length:start + 2 * length]
            if span == next_span:
                # Check if this repetition is legal (exists in ref at same frequency)
                ref_count = ref.count(span)
                hyp_count = hyp.count(span)
                excess = hyp_count - ref_count
                if excess > 0:
                    repeats.append({
                        "span": span,
                        "length": length,
                        "excess_count": excess,
                    })

    # Deduplicate: keep longest spans
    unique_repeats = {}
    for r in repeats:
        key = r["span"]
        if key not in unique_repeats or r["length"] > unique_repeats[key]["length"]:
            unique_repeats[key] = r

    repeat_list = list(unique_repeats.values())

    return {
        "repeat_span_count": len(repeat_list),
        "longest_repeat_span": max((r["length"] for r in repeat_list), default=0),
        "repeated_char_count": sum(r["length"] * r["excess_count"] for r in repeat_list),
    }


# === Status determination ===

def determine_status(metrics: Dict, teacher_baseline: Optional[Dict] = None) -> Tuple[str, List[str]]:
    """Determine PASS/WARN/FAIL with reasons."""
    reasons = []

    # Hard fails
    if metrics["trailing_deletion_ratio"] > 0.15:
        reasons.append("EARLY_STOP")

    if metrics["complete_lines"] < metrics["total_lines"]:
        missing = metrics["total_lines"] - metrics["complete_lines"]
        if missing > 0:
            reasons.append("MISSING_LINE")

    if metrics["repeat"]["longest_repeat_span"] >= 4:
        reasons.append("REPEATED_SPAN")

    if metrics["audio_duration"] < 1.0:
        reasons.append("CORRUPT_AUDIO")

    # Warns
    if teacher_baseline:
        if metrics["cer"] > teacher_baseline.get("median_cer", 0.05) + 0.03:
            reasons.append("HIGH_CER")

    dur_ratio = metrics.get("duration_ratio")
    if dur_ratio is not None and (dur_ratio < 0.7 or dur_ratio > 1.8):
        reasons.append("ABNORMAL_DURATION")

    if any(r in reasons for r in ["EARLY_STOP", "MISSING_LINE", "REPEATED_SPAN", "CORRUPT_AUDIO"]):
        return "FAIL", reasons
    elif reasons:
        return "WARN", reasons
    else:
        return "PASS", []


# === ASR ===

def load_asr_model():
    """Load FunASR Paraformer-large for Chinese."""
    from funasr import AutoModel
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device="cpu",
        disable_update=True,
    )
    return model


def transcribe(asr_model, wav_path: str) -> Tuple[str, float]:
    """Return (text, duration_seconds)."""
    import soundfile as sf
    info = sf.info(wav_path)
    duration = info.frames / info.samplerate

    result = asr_model.generate(
        input=wav_path,
        batch_size_s=300,
    )
    text = ""
    if result and len(result) > 0:
        text = result[0].get("text", "")

    return text, duration


# === Main evaluation ===

def evaluate(
    test_set_path: str,
    audio_dir: str,
    model_version: str,
    teacher_audio_dir: Optional[str],
    output_csv: str,
):
    with open(test_set_path, encoding="utf-8") as f:
        test_set = json.load(f)

    samples = test_set["samples"]
    audio_path = Path(audio_dir)
    teacher_path = Path(teacher_audio_dir) if teacher_audio_dir else None

    print(f"Loading ASR model...")
    asr_model = load_asr_model()
    print(f"ASR loaded.")

    # Pre-normalize reference texts
    for s in samples:
        s["_norm_content"] = normalize_text(s["content"])
        s["_norm_lines"] = [normalize_text(l) for l in s["text_lines"]]
        # Build title+author prefix for stripping from ASR output
        prefix = s.get("title", "") + s.get("dynasty", "") + s.get("author", "")
        s["_norm_prefix"] = normalize_text(prefix)

    results = []

    # Phase 1: Evaluate teacher audio if provided (to establish baseline)
    teacher_results = []
    if teacher_path:
        print(f"\n=== Phase 1: Teacher baseline ===")
        for s in samples:
            teacher_wav_name = s.get("teacher_wav")
            if not teacher_wav_name:
                continue
            teacher_wav = teacher_path / teacher_wav_name
            if not teacher_wav.exists():
                print(f"  [SKIP] {s['poem_id']}: teacher wav not found at {teacher_wav.name}")
                continue

            print(f"  [TEACHER] {s['title']}...", end=" ")
            asr_text, dur = transcribe(asr_model, str(teacher_wav))
            norm_asr = normalize_text(asr_text)
            norm_asr = strip_prefix(norm_asr, s.get("_norm_prefix", ""))

            cer_metrics = compute_cer(s["_norm_content"], norm_asr)
            line_cov = compute_line_coverage(s["_norm_lines"], norm_asr)
            complete_lines = sum(1 for c in line_cov if c >= 0.5)
            rep = detect_repeats(s["_norm_content"], norm_asr)

            print(f"CER={cer_metrics['cer']:.1%}")

            teacher_results.append({
                "poem_id": s["poem_id"],
                "cer": cer_metrics["cer"],
                "asr_text": norm_asr[:60],
            })

    # Compute teacher baseline
    teacher_baseline = None
    if teacher_results:
        import statistics
        certs = [r["cer"] for r in teacher_results]
        teacher_baseline = {
            "median_cer": statistics.median(certs),
            "p95_cer": sorted(certs)[int(len(certs) * 0.95)] if len(certs) > 1 else certs[0],
        }
        print(f"\nTeacher baseline: median_CER={teacher_baseline['median_cer']:.1%}, P95_CER={teacher_baseline['p95_cer']:.1%}")

    # Phase 2: Evaluate student audio
    print(f"\n=== Phase 2: {model_version} ===")
    for s in samples:
        # Find audio file
        wav = audio_path / f"{s['poem_id']}_{model_version}.wav"
        if not wav.exists():
            # Try alternative naming
            wav = audio_path / f"{s['poem_id']}.wav"
        if not wav.exists():
            print(f"  [SKIP] {s['poem_id']}: audio not found")
            continue

        sid = s["poem_id"]
        print(f"  [{sid}] {s['title']}...", end=" ")

        asr_text, dur = transcribe(asr_model, str(wav))
        norm_asr = normalize_text(asr_text)
        norm_asr = strip_prefix(norm_asr, s.get("_norm_prefix", ""))

        cer_metrics = compute_cer(s["_norm_content"], norm_asr)
        line_cov = compute_line_coverage(s["_norm_lines"], norm_asr)
        complete_lines = sum(1 for c in line_cov if c >= 0.5)
        rep = detect_repeats(s["_norm_content"], norm_asr)

        # Teacher duration for ratio
        teacher_dur = None
        if teacher_path:
            teacher_wav_name = s.get("teacher_wav")
            if teacher_wav_name:
                teacher_wav = teacher_path / teacher_wav_name
                if teacher_wav.exists():
                    import torchaudio
                    t_info = torchaudio.info(str(teacher_wav))
                    teacher_dur = t_info.num_frames / t_info.sample_rate

        dur_ratio = dur / teacher_dur if teacher_dur and teacher_dur > 0 else None

        metrics = {
            "poem_id": sid,
            "title": s["title"],
            "form": s["form"],
            "split": s["split"],
            "token_length": len(s["_norm_content"]),
            "audio_duration": round(dur, 2),
            "teacher_duration": round(teacher_dur, 2) if teacher_dur else None,
            "duration_ratio": round(dur_ratio, 3) if dur_ratio else None,
            "cer": cer_metrics["cer"],
            "substitution_rate": cer_metrics["substitution_rate"],
            "deletion_rate": cer_metrics["deletion_rate"],
            "insertion_rate": cer_metrics["insertion_rate"],
            "trailing_deletion": cer_metrics["trailing_deletion"],
            "trailing_deletion_ratio": cer_metrics["trailing_deletion_ratio"],
            "complete_lines": complete_lines,
            "total_lines": len(s["text_lines"]),
            "line_coverages": line_cov,
            "repeat": rep,
            "raw_asr_text": asr_text[:100],
            "normalized_asr_text": norm_asr[:100],
        }

        status, reasons = determine_status(metrics, teacher_baseline)
        metrics["status"] = status
        metrics["failure_reasons"] = reasons

        print(f"CER={cer_metrics['cer']:.1%} D={cer_metrics['deletion_rate']:.1%} I={cer_metrics['insertion_rate']:.1%} lines={complete_lines}/{len(s['text_lines'])} [{status}]")

        results.append(metrics)

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary ({model_version})")
    print(f"{'='*80}")
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    print(f"  PASS: {pass_count}, WARN: {warn_count}, FAIL: {fail_count}")

    if fail_count > 0 or warn_count > 0:
        print(f"\n  Issues:")
        for r in results:
            if r["status"] != "PASS":
                print(f"    [{r['status']}] {r['poem_id']} ({r['form']}): {', '.join(r['failure_reasons'])}")

    # Write CSV
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "poem_id", "title", "form", "split", "token_length",
        "audio_duration", "teacher_duration", "duration_ratio",
        "cer", "substitution_rate", "deletion_rate", "insertion_rate",
        "trailing_deletion", "trailing_deletion_ratio",
        "complete_lines", "total_lines",
        "repeat_span_count", "longest_repeat_span",
        "status", "failure_reasons",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: v for k, v in r.items() if k in fieldnames}
            row["failure_reasons"] = "; ".join(r.get("failure_reasons", []))
            if "longest_repeat_span" not in row and "repeat" in r:
                row["longest_repeat_span"] = r["repeat"]["longest_repeat_span"]
            if "repeat_span_count" not in row and "repeat" in r:
                row["repeat_span_count"] = r["repeat"]["repeat_span_count"]
            writer.writerow(row)

    print(f"\n  Report saved to: {csv_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate TTS content correctness")
    parser.add_argument("--test-set", default="data/test_set_v1.json")
    parser.add_argument("--audio-dir", required=True, help="Directory with student WAV files")
    parser.add_argument("--model-version", required=True, help="Label for audio filename suffix")
    parser.add_argument("--teacher-audio-dir", default=None, help="Directory with teacher WAV files")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    output_csv = args.output_csv or f"data/eval_{args.model_version}.csv"

    evaluate(
        test_set_path=args.test_set,
        audio_dir=args.audio_dir,
        model_version=args.model_version,
        teacher_audio_dir=args.teacher_audio_dir,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()
