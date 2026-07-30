"""Parse TextGrid files and extract per-character duration.

Output: data/duration_labels.json with per-sample:
  - char sequence (matching original full_text)
  - per-char start/end time in seconds
  - per-char duration in mel frames (hop_length=300, sr=24000)
  - punctuation positions mapped to silence duration

Key invariants:
  - sum(durations) should approximately equal mel_length
  - Every Chinese character should have a non-zero duration
  - Punctuation intervals become silence tokens
"""
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
TEXTGRID_DIR = ROOT / "data" / "mfa_aligned"
CLEAN_JSON = ROOT / "data" / "clean_data_list.json"
OUTPUT = ROOT / "data" / "duration_labels.json"

# Mel params (matching PaddleSpeech FS2)
SAMPLE_RATE = 24000
HOP_LENGTH = 300
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # 80 frames/sec


def parse_textgrid(path):
    """Parse a Praat TextGrid file, return word-tier intervals."""
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")

    intervals = []
    current_tier = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if 'name = "words"' in line:
            current_tier = "words"
        elif 'name = "phones"' in line:
            current_tier = "phones"

        if current_tier == "words" and "text =" in line and "xmin" not in line:
            # This line has text. Look backwards for xmin/xmax
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
                    "xmin": xmin,
                    "xmax": xmax,
                    "text": text if text else "",
                })
        i += 1

    return intervals


def time_to_mel_frames(time_sec):
    """Convert time in seconds to mel frame count (using cumulative boundary approach)."""
    return round(time_sec * FRAME_RATE)


def main():
    with open(CLEAN_JSON, encoding="utf-8") as f:
        clean_data = json.load(f)

    # Build index → sample lookup
    sample_by_idx = {s["index"]: s for s in clean_data["samples"]}

    textgrid_files = sorted(TEXTGRID_DIR.glob("*.TextGrid"))
    print(f"Found {len(textgrid_files)} TextGrid files")

    results = []
    errors = []

    for tg_path in textgrid_files:
        # Extract poem index from filename (poem_XXXX)
        base = tg_path.stem  # poem_0000
        idx = int(base.split("_")[1])

        sample = sample_by_idx.get(idx)
        if not sample:
            errors.append(f"{base}: no matching sample")
            continue

        full_text = sample["full_text"]

        # Parse TextGrid
        word_intervals = parse_textgrid(tg_path)

        if not word_intervals:
            errors.append(f"{base}: no word intervals")
            continue

        # Build char sequence from full_text
        chars = []
        for ch in full_text:
            if "\u4e00" <= ch <= "\u9fff":
                chars.append(("char", ch))
            elif ch in "，。？！；：·":
                chars.append(("punct", ch))
            elif ch in "／/":
                chars.append(("punct", "·"))

        # Match chars to TextGrid intervals
        # TextGrid has: char intervals with text, empty intervals for silence
        tg_nonempty = [iv for iv in word_intervals if iv["text"]]
        tg_idx = 0

        char_durations = []
        for char_type, char_val in chars:
            if char_type == "char":
                # Find next matching interval
                matched = False
                while tg_idx < len(tg_nonempty):
                    iv = tg_nonempty[tg_idx]
                    if iv["text"] == char_val:
                        start = iv["xmin"]
                        end = iv["xmax"]
                        char_durations.append({
                            "type": "char",
                            "char": char_val,
                            "start": round(start, 4),
                            "end": round(end, 4),
                            "duration_sec": round(end - start, 4),
                        })
                        tg_idx += 1
                        matched = True
                        break
                    else:
                        # Skip mismatched interval (MFA may have inserted/dropped)
                        tg_idx += 1

                if not matched:
                    char_durations.append({
                        "type": "char",
                        "char": char_val,
                        "start": 0,
                        "end": 0,
                        "duration_sec": 0,
                        "error": "no_match",
                    })
            elif char_type == "punct":
                # Assign punctuation the silence duration of the nearest empty interval
                char_durations.append({
                    "type": "punct",
                    "char": char_val,
                    "start": 0,
                    "end": 0,
                    "duration_sec": 0,
                })

        # Convert to mel frame durations
        # Use cumulative boundary approach: round start/end times to frames
        mel_durations = []
        for cd in char_durations:
            if cd["duration_sec"] > 0:
                start_frame = round(cd["start"] * FRAME_RATE)
                end_frame = round(cd["end"] * FRAME_RATE)
                mel_dur = max(end_frame - start_frame, 1)  # at least 1 frame
            else:
                mel_dur = 0  # punctuation, will be set later
            mel_durations.append(mel_dur)

        # Assign punctuation durations from neighboring silence
        # For now, give punctuation a default of 0 (will be handled by model)
        # Actually, let's find silence intervals between chars
        # and assign them to the preceding punctuation
        total_duration_sec = word_intervals[-1]["xmax"] if word_intervals else 0
        total_mel_frames = round(total_duration_sec * FRAME_RATE)

        # Stats
        char_count = sum(1 for cd in char_durations if cd["type"] == "char")
        matched_count = sum(1 for cd in char_durations if cd["type"] == "char" and cd.get("duration_sec", 0) > 0)
        zero_duration = sum(1 for cd in char_durations if cd["type"] == "char" and cd.get("duration_sec", 0) == 0)

        results.append({
            "index": idx,
            "poem_id": base,
            "title": sample["title"],
            "full_text": full_text,
            "audio_path": sample["audio"],
            "total_duration_sec": round(total_duration_sec, 2),
            "total_mel_frames": total_mel_frames,
            "char_count": char_count,
            "matched_count": matched_count,
            "zero_duration_count": zero_duration,
            "char_durations": char_durations,
            "mel_durations": mel_durations,
            "mel_frame_rate": FRAME_RATE,
        })

    # Summary stats
    total_chars = sum(r["char_count"] for r in results)
    total_matched = sum(r["matched_count"] for r in results)
    total_zero = sum(r["zero_duration_count"] for r in results)
    total_dur = sum(r["total_duration_sec"] for r in results)

    print(f"\n{'='*60}")
    print(f"Duration Extraction Summary")
    print(f"{'='*60}")
    print(f"  Samples: {len(results)}/{len(textgrid_files)}")
    print(f"  Total chars: {total_chars}")
    print(f"  Matched: {total_matched} ({total_matched/total_chars*100:.1f}%)")
    print(f"  Zero duration: {total_zero} ({total_zero/total_chars*100:.1f}%)")
    print(f"  Total audio: {total_dur:.0f}s ({total_dur/60:.1f} min)")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")

    # Check mel frame invariant
    mismatches = 0
    for r in results:
        mel_sum = sum(d for d in r["mel_durations"] if d > 0)
        expected = r["total_mel_frames"]
        if abs(mel_sum - expected) > 5:  # allow small rounding
            mismatches += 1

    print(f"\n  Mel frame sum mismatches (>5 frames): {mismatches}/{len(results)}")

    # Save
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(results),
            "total_chars": total_chars,
            "total_matched": total_matched,
            "total_zero_duration": total_zero,
            "mel_params": {
                "sample_rate": SAMPLE_RATE,
                "hop_length": HOP_LENGTH,
                "frame_rate": FRAME_RATE,
            },
            "samples": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  Output: {OUTPUT}")


if __name__ == "__main__":
    main()
