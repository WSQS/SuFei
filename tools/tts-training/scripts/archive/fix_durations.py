"""Fix build_durations: scale TextGrid timestamps to match mel duration.

The root cause: TextGrid xmax (11.92s) != audio duration (8.575s) != mel
duration (687/80 = 8.588s). MFA generated TextGrids with incorrect total
duration, possibly due to a different audio version or MFA internal padding.

Fix: Instead of using absolute TextGrid timestamps, compute durations as
proportional allocations of the actual mel_len, based on TextGrid interval
proportions. Then assign per-character durations proportionally.

This is a linear rescaling: scale_factor = mel_len / textgrid_total_frames
"""
import sys, json, math
import numpy as np
from pathlib import Path

sys.path.insert(0, r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training\scripts")
from generate_paddlespeech_distillation_data import (
    parse_textgrid_intervals, text_to_phonemes, FRAME_RATE, INITIAL_RATIO,
    load_phone_id_map, phones_to_ids,
)

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
FEAT_DIR = ROOT / "data" / "paddle_distill_features"


def build_durations_fixed(phonemes, tg_intervals, mel_len):
    """Build per-phoneme durations from TextGrid, scaled to mel_len.

    Instead of using absolute timestamps (which may not match mel), we:
    1. Compute per-character durations from TextGrid intervals (in seconds)
    2. Scale them proportionally to fill mel_len
    3. Split into initial/final for multi-phoneme characters
    """
    tg_total = tg_intervals[-1]["xmax"] if tg_intervals else 1.0
    
    # Build timeline in seconds
    timeline = []
    for iv in tg_intervals:
        timeline.append({
            "text": iv["text"],
            "xmin": iv["xmin"],
            "xmax": iv["xmax"],
            "dur_s": iv["xmax"] - iv["xmin"],
            "is_silence": iv["is_silence"],
        })
    
    durations_mel = []
    tg_idx = 0
    n_phonemes = len(phonemes)
    
    while tg_idx < len(timeline) or len(durations_mel) < n_phonemes:
        # Process next phoneme
        if len(durations_mel) >= n_phonemes:
            break
        
        i = len(durations_mel)
        phone_str, char = phonemes[i]
        
        if phone_str == "<eos>":
            durations_mel.append(0)  # will be filled by proportional scaling
            continue
        
        if phone_str in ("\uff0c", "\u3002", "\uff1f", "\uff01"):
            # Punctuation: find next silence interval
            silence_dur_s = 0.0
            while tg_idx < len(timeline):
                tl = timeline[tg_idx]
                if tl["is_silence"]:
                    silence_dur_s = tl["dur_s"]
                    tg_idx += 1
                    break
                else:
                    tg_idx += 1
            durations_mel.append(silence_dur_s)  # in seconds, will scale later
            continue
        
        # Find matching character interval
        char_dur_s = 0.0
        while tg_idx < len(timeline):
            tl = timeline[tg_idx]
            if tl["text"] == char and not tl["is_silence"]:
                char_dur_s = tl["dur_s"]
                tg_idx += 1
                break
            else:
                tg_idx += 1
        
        if char_dur_s == 0.0:
            # Character not found, assign small default
            char_dur_s = 0.05  # 50ms default
        
        if i + 1 < n_phonemes and phonemes[i + 1][1] == char:
            # Multi-phoneme character (initial + final)
            init_dur_s = char_dur_s * INITIAL_RATIO
            final_dur_s = char_dur_s - init_dur_s
            durations_mel.append(init_dur_s)
            durations_mel.append(final_dur_s)
        else:
            durations_mel.append(char_dur_s)
    
    # Now durations_mel is in seconds. Scale to mel frames.
    # Remove <eos> entries for scaling
    eos_indices = [i for i, p in enumerate(phonemes) if p[0] == "<eos>"]
    scale_targets = [d for i, d in enumerate(durations_mel) if i not in eos_indices]
    
    total_s = sum(scale_targets)
    if total_s > 0:
        scale_factor = mel_len / (total_s * FRAME_RATE)
    else:
        scale_factor = 1.0
    
    # Scale and round
    result = []
    scale_idx = 0
    for i, phone in enumerate(phonemes):
        if phone[0] == "<eos>":
            result.append(1)
        else:
            scaled = durations_mel[i] * scale_factor * FRAME_RATE
            result.append(max(round(scaled), 1))
    
    # Fix sum to match mel_len exactly
    diff = mel_len - sum(result)
    if diff != 0:
        max_idx = max(range(len(result)), key=lambda x: result[x])
        result[max_idx] = max(result[max_idx] + diff, 1)
    
    return result


# Test on a few poems
manifest = [json.loads(l) for l in open(ROOT / "data" / "train_300_manifest.jsonl", encoding="utf-8")]

print("Testing fixed duration builder on 5 poems:\n")
for rec in manifest[:5]:
    pid = rec["poem_id"]
    mel_len = rec["mel_len"]
    text = rec["text"]
    phonemes = text_to_phonemes(text)
    
    tg_path = ROOT / "data" / "mfa_aligned" / f"{pid}.TextGrid"
    if not tg_path.exists():
        print(f"  {pid}: no TextGrid")
        continue
    
    intervals = parse_textgrid_intervals(tg_path)
    durations = build_durations_fixed(phonemes, intervals, mel_len)
    
    old_durations = rec["durations"]
    
    # Stats
    n_phones = len(durations)
    dur_arr = np.array(durations)
    old_arr = np.array(old_durations)
    
    print(f"{pid}: {text[:35]}")
    print(f"  mel_len={mel_len}, n_phones={n_phones}, sum={sum(durations)}")
    print(f"  Old: first={old_arr[0]} ({old_arr[0]/sum(old_arr)*100:.0f}%), "
          f"n_default2={np.sum(old_arr[1:]==2)}/{len(old_arr)-1}")
    print(f"  New: first={dur_arr[0]} ({dur_arr[0]/sum(dur_arr)*100:.0f}%), "
          f"min={dur_arr.min()} max={dur_arr.max()} mean={dur_arr.mean():.1f} std={dur_arr.std():.1f}")
    print(f"  New first 10: {durations[:10]}")
    print(f"  Old first 10: {old_durations[:10]}")
    print()

# Run on ALL poems to check statistics
print("="*70)
print("Full dataset statistics:")
print("="*70)

all_first_durs = []
all_non_first = []
for rec in manifest:
    pid = rec["poem_id"]
    mel_len = rec["mel_len"]
    phonemes = text_to_phonemes(rec["text"])
    tg_path = ROOT / "data" / "mfa_aligned" / f"{pid}.TextGrid"
    if not tg_path.exists():
        continue
    intervals = parse_textgrid_intervals(tg_path)
    durations = build_durations_fixed(phonemes, intervals, mel_len)
    
    all_first_durs.append(durations[0])
    all_non_first.extend(durations[1:])

all_first_durs = np.array(all_first_durs)
all_non_first = np.array(all_non_first)

print(f"\nFirst phoneme: mean={all_first_durs.mean():.1f} median={np.median(all_first_durs):.0f} "
      f"min={all_first_durs.min()} max={all_first_durs.max()}")
first_pcts = []
for rec in manifest:
    pid = rec["poem_id"]
    tg_path = ROOT / "data" / "mfa_aligned" / f"{pid}.TextGrid"
    if not tg_path.exists():
        continue
    intervals = parse_textgrid_intervals(tg_path)
    durations = build_durations_fixed(text_to_phonemes(rec["text"]), intervals, rec["mel_len"])
    first_pcts.append(durations[0] / sum(durations) * 100)

print(f"First phoneme % of total: mean={np.mean(first_pcts):.0f}%")

print(f"\nNon-first: mean={all_non_first.mean():.1f} median={np.median(all_non_first):.0f} "
      f"min={all_non_first.min()} max={all_non_first.max()}")
print(f"  =2: {np.sum(all_non_first==2)}/{len(all_non_first)} ({np.sum(all_non_first==2)/len(all_non_first)*100:.1f}%)")
print(f"  Distribution: {dict(list(sorted(__import__('collections').Counter(all_non_first.tolist()).items()))[:10])}")
