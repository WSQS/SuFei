"""Build FastSpeech 2 training manifest (JSONL).

Per-sample record:
  {
    "poem_id": "poem_0000",
    "phoneme_ids": [37, 82, 2, ...],   # pinyin-based phone IDs
    "durations": [3, 7, 12, ...],       # per-phoneme mel frames, sum == mel_length
    "mel_path": "data/nar_features/poem_0000.npz",
    "mel_len": 938,
    "text": "江南曲，唐代·李益。..."
  }

Duration allocation strategy:
  - Char with initial+final: initial gets 30%, final gets 70% (min 1 each)
  - Char without initial (zero-initial): full duration to the single final
  - Punctuation: assigned from TextGrid silence intervals
  - Trailing gap: distributed to the last token to ensure sum == mel_length
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DURATION_JSON = ROOT / "data" / "duration_labels.json"
PHONE_ID_MAP = ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0" / "phone_id_map.txt"
TEXTGRID_DIR = ROOT / "data" / "mfa_aligned"
OUTPUT = ROOT / "data" / "train_manifest.jsonl"

# Mel params
SAMPLE_RATE = 24000
HOP_LENGTH = 300
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # 80

# Duration split: initial vs final
INITIAL_RATIO = 0.3

# PaddleSpeech initials
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
    with open(PHONE_ID_MAP, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                phone_map[parts[0]] = int(parts[1])
    return phone_map


def parse_textgrid_intervals(path):
    """Parse TextGrid, return all word-tier intervals (including silence)."""
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
                    "xmin": xmin,
                    "xmax": xmax,
                    "text": text,
                    "is_silence": len(text) == 0,
                })
        i += 1

    return intervals


def text_to_phonemes(text):
    """Convert Chinese text to PaddleSpeech-style phoneme sequence.

    Returns list of (phoneme_str, char_or_punct_marker).
    """
    from pypinyin import pinyin, Style

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

            # Fix pypinyin-specific finals
            for old, new in PYPINYIN_FINAL_FIX.items():
                if py_base == old:
                    py_base = new
                    break

            # Try to split into initial + final
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

        elif char in PUNCT_TO_PHONE:
            result.append((PUNCT_TO_PHONE[char], char))

    result.append(("<eos>", None))
    return result


def build_durations(phonemes, tg_intervals, char_durations):
    """Build per-phoneme duration array that sums to total mel length.

    Strategy:
    1. Walk TextGrid intervals to get per-segment timing (char + silence)
    2. For char phonemes: split duration between initial and final
    3. For punctuation: assign nearest preceding silence interval duration
    4. Fix remainder on last phoneme
    """
    total_time = tg_intervals[-1]["xmax"] if tg_intervals else 0
    total_mel = round(total_time * FRAME_RATE)

    # Build a timeline of (text, start_frame, end_frame) from TextGrid
    timeline = []
    for iv in tg_intervals:
        start_f = round(iv["xmin"] * FRAME_RATE)
        end_f = round(iv["xmax"] * FRAME_RATE)
        dur = max(end_f - start_f, 0)
        timeline.append({
            "text": iv["text"],
            "start": start_f,
            "end": end_f,
            "dur": dur,
            "is_silence": iv["is_silence"],
        })

    # Walk phonemes and assign durations
    durations = []
    tg_idx = 0  # position in timeline

    # Group phonemes by char
    # phonemes is list of (phone_str, char_or_None)
    # Consecutive phonemes with same char belong together

    i = 0
    while i < len(phonemes):
        phone_str, char = phonemes[i]

        if phone_str == "<eos>":
            durations.append(1)
            i += 1
            continue

        # Check if punctuation
        if phone_str in ("，", "。", "？", "！"):
            # Find nearest silence interval at or after current tg position
            silence_dur = 0
            while tg_idx < len(timeline):
                tl = timeline[tg_idx]
                if tl["is_silence"]:
                    silence_dur = tl["dur"]
                    tg_idx += 1
                    break
                elif tl["text"]:
                    # Non-silence, non-empty — skip chars we've passed
                    tg_idx += 1
                else:
                    tg_idx += 1
            durations.append(max(silence_dur, 1))
            i += 1
            continue

        # It's a char phoneme. Find matching interval in timeline
        # Advance tg_idx to find the char
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

        # Check if next phoneme is the final (same char)
        if i + 1 < len(phonemes) and phonemes[i + 1][1] == char:
            # initial + final
            init_dur = max(round(char_dur * INITIAL_RATIO), 1)
            final_dur = max(char_dur - init_dur, 1)
            durations.append(init_dur)
            durations.append(final_dur)
            i += 2
        else:
            # single phoneme (no initial)
            durations.append(max(char_dur, 1))
            i += 1

    # Fix sum to match total_mel
    current_sum = sum(durations)
    diff = total_mel - current_sum
    if diff != 0:
        # Add diff to the phoneme with largest duration (usually a final or silence)
        max_idx = max(range(len(durations) - 1), key=lambda x: durations[x])
        durations[max_idx] = max(durations[max_idx] + diff, 1)

    return durations


def main():
    phone_map = load_phone_id_map()
    print(f"Phone vocab: {len(phone_map)} entries")

    with open(DURATION_JSON, encoding="utf-8") as f:
        dur_data = json.load(f)

    samples = dur_data["samples"]
    print(f"Processing {len(samples)} samples...")

    records = []
    errors = []
    missing_phones = set()

    for s in samples:
        poem_id = s["poem_id"]
        full_text = s["full_text"]
        char_durations = s["char_durations"]

        # Parse TextGrid for silence info
        tg_path = TEXTGRID_DIR / f"{poem_id}.TextGrid"
        if not tg_path.exists():
            errors.append(f"{poem_id}: no TextGrid")
            continue
        tg_intervals = parse_textgrid_intervals(tg_path)

        # G2P
        phonemes = text_to_phonemes(full_text)

        # Convert to IDs
        phone_ids = []
        for pstr, _ in phonemes:
            if pstr in phone_map:
                phone_ids.append(phone_map[pstr])
            else:
                missing_phones.add(pstr)
                phone_ids.append(phone_map.get("<unk>", 1))

        # Build durations
        durations = build_durations(phonemes, tg_intervals, char_durations)

        # Load npz to get actual mel length
        npz_path = ROOT / "data" / "nar_features" / f"{poem_id}.npz"
        if not npz_path.exists():
            errors.append(f"{poem_id}: no features")
            continue

        npz = np.load(str(npz_path))
        mel_len = npz["mel"].shape[0]

        # Verify and fix duration sum to match actual mel_len
        dur_sum = sum(durations)
        if dur_sum != mel_len:
            diff = mel_len - dur_sum
            max_idx = max(range(len(durations)), key=lambda x: durations[x])
            durations[max_idx] = max(durations[max_idx] + diff, 1)

        # Final check
        assert sum(durations) == mel_len, f"{poem_id}: dur sum {sum(durations)} != mel_len {mel_len}"

        records.append({
            "poem_id": poem_id,
            "phoneme_ids": phone_ids,
            "durations": durations,
            "mel_path": f"data/nar_features/{poem_id}.npz",
            "mel_len": mel_len,
            "n_phonemes": len(phone_ids),
            "text": full_text,
        })

    # Write JSONL
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    print(f"\n{'='*60}")
    print(f"Manifest built: {OUTPUT}")
    print(f"  Records: {len(records)}")
    print(f"  Errors: {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"    {e}")

    if missing_phones:
        print(f"  Missing phones ({len(missing_phones)}): {sorted(missing_phones)[:20]}")

    total_phonemes = sum(r["n_phonemes"] for r in records)
    total_mel = sum(r["mel_len"] for r in records)
    print(f"  Total phonemes: {total_phonemes}")
    print(f"  Total mel frames: {total_mel} ({total_mel / FRAME_RATE / 60:.1f} min)")

    # Verify invariant on all records
    bad = 0
    for r in records:
        if sum(r["durations"]) != r["mel_len"]:
            bad += 1
    print(f"  Duration invariant violations: {bad}/{len(records)}")

    # Sample preview
    if records:
        r = records[0]
        print(f"\n  Sample {r['poem_id']}:")
        print(f"    text: {r['text'][:40]}")
        print(f"    n_phonemes: {r['n_phonemes']}")
        print(f"    mel_len: {r['mel_len']}")
        print(f"    dur_sum: {sum(r['durations'])}")
        print(f"    first 10 ids: {r['phoneme_ids'][:10]}")
        print(f"    first 10 durs: {r['durations'][:10]}")


if __name__ == "__main__":
    main()
