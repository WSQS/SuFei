"""Deep analysis: why are 91% of durations = 2 frames?

Investigate the duration fix pipeline:
1. Full dataset duration distribution
2. Sample TextGrid intervals to understand alignment
3. Trace build_durations_fixed step by step on real examples
4. Check if the scaling math is correct
"""
import sys, json, math, numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training\scripts")
from generate_paddlespeech_distillation_data import (
    parse_textgrid_intervals, text_to_phonemes, FRAME_RATE, INITIAL_RATIO,
)
# Inline build_durations_fixed to avoid import path issues
def build_durations_fixed(phonemes, tg_intervals, mel_len):
    tg_total = tg_intervals[-1]["xmax"] if tg_intervals else 1.0
    timeline = [{"text": iv["text"], "xmin": iv["xmin"], "xmax": iv["xmax"],
                 "dur_s": iv["xmax"] - iv["xmin"], "is_silence": iv["is_silence"]}
                for iv in tg_intervals]
    durations_mel = []
    tg_idx = 0
    n_phonemes = len(phonemes)
    while tg_idx < len(timeline) or len(durations_mel) < n_phonemes:
        if len(durations_mel) >= n_phonemes:
            break
        i = len(durations_mel)
        phone_str, char = phonemes[i]
        if phone_str == "<eos>":
            durations_mel.append(0)
            continue
        if phone_str in ("\uff0c", "\u3002", "\uff1f", "\uff01"):
            silence_dur_s = 0.0
            while tg_idx < len(timeline):
                tl = timeline[tg_idx]
                if tl["is_silence"]:
                    silence_dur_s = tl["dur_s"]
                    tg_idx += 1
                    break
                else:
                    tg_idx += 1
            durations_mel.append(silence_dur_s)
            continue
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
            char_dur_s = 0.05
        if i + 1 < n_phonemes and phonemes[i + 1][1] == char:
            init_dur_s = char_dur_s * INITIAL_RATIO
            final_dur_s = char_dur_s - init_dur_s
            durations_mel.append(init_dur_s)
            durations_mel.append(final_dur_s)
        else:
            durations_mel.append(char_dur_s)
    eos_indices = [i for i, p in enumerate(phonemes) if p[0] == "<eos>"]
    scale_targets = [d for i, d in enumerate(durations_mel) if i not in eos_indices]
    total_s = sum(scale_targets)
    if total_s > 0:
        scale_factor = mel_len / (total_s * FRAME_RATE)
    else:
        scale_factor = 1.0
    result = []
    for i, phone in enumerate(phonemes):
        if phone[0] == "<eos>":
            result.append(1)
        else:
            scaled = durations_mel[i] * scale_factor * FRAME_RATE
            result.append(max(round(scaled), 1))
    diff = mel_len - sum(result)
    if diff != 0:
        max_idx = max(range(len(result)), key=lambda x: result[x])
        result[max_idx] = max(result[max_idx] + diff, 1)
    return result

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")

manifest = [json.loads(l) for l in open(ROOT / "data" / "train_300_manifest.jsonl", encoding="utf-8")]

# ============================================================================
# 1. Full dataset duration distribution
# ============================================================================
print("="*70)
print("1. FULL DATASET DURATION DISTRIBUTION")
print("="*70)

all_durs = []
first_durs = []
for rec in manifest:
    durs = rec["durations"]
    all_durs.extend(durs)
    first_durs.append(durs[0])

all_durs = np.array(all_durs)
print(f"Total phonemes: {len(all_durs)}")
print(f"Mean: {all_durs.mean():.2f}  Median: {np.median(all_durs):.0f}  Std: {all_durs.std():.2f}")
print(f"Distribution:")
c = Counter(all_durs.tolist())
for k in sorted(c.keys())[:15]:
    pct = c[k] / len(all_durs) * 100
    print(f"  dur={k:>3}: {c[k]:>6} ({pct:>5.1f}%)")

# ============================================================================
# 2. Detailed trace on 3 examples
# ============================================================================
print(f"\n{'='*70}")
print("2. DETAILED TRACE: build_durations_fixed")
print("="*70)

for rec in manifest[:3]:
    pid = rec["poem_id"]
    mel_len = rec["mel_len"]
    text = rec["text"]
    
    tg_path = ROOT / "data" / "mfa_aligned" / f"{pid}.TextGrid"
    if not tg_path.exists():
        print(f"  {pid}: no TextGrid")
        continue
    
    intervals = parse_textgrid_intervals(tg_path)
    phonemes = text_to_phonemes(text)
    
    print(f"\n--- {pid}: {text[:40]} ---")
    print(f"  mel_len={mel_len}, n_phonemes={len(phonemes)}")
    print(f"  TextGrid intervals ({len(intervals)} total):")
    tg_total = intervals[-1]["xmax"] if intervals else 0
    print(f"  TextGrid xmax={tg_total:.3f}s, mel_time={mel_len/FRAME_RATE:.3f}s")
    print(f"  scale_factor = mel_len / (tg_total * FRAME_RATE) = {mel_len / (tg_total * FRAME_RATE):.3f}")
    
    # Show first 15 intervals
    for iv in intervals[:15]:
        dur_s = iv["xmax"] - iv["xmin"]
        label = iv["text"] if iv["text"] else "(silence)"
        print(f"    [{iv['xmin']:.3f}-{iv['xmax']:.3f}] ({dur_s:.3f}s) '{label}'")
    
    # Show phonemes
    print(f"  Phonemes ({len(phonemes)}):")
    phoneme_strs = [p[0] for p in phonemes]
    for i, (ps, char) in enumerate(phonemes[:15]):
        print(f"    [{i}] {ps} (char='{char}')")
    
    # Build durations
    durations = build_durations_fixed(phonemes, intervals, mel_len)
    print(f"  Result durations ({len(durations)}): {durations[:20]}")
    print(f"  Sum={sum(durations)} vs mel_len={mel_len}")
    
    # Trace the matching: does the TextGrid character match?
    print(f"\n  Matching trace:")
    tg_idx = 0
    timeline = [{"text": iv["text"], "xmin": iv["xmin"], "xmax": iv["xmax"],
                 "dur_s": iv["xmax"] - iv["xmin"], "is_silence": iv["is_silence"]}
                for iv in intervals]
    
    for i, (phone_str, char) in enumerate(phonemes[:10]):
        if phone_str == "<eos>":
            print(f"    phoneme[{i}] = <eos> -> dur=0 (will be 1)")
            continue
        if phone_str in ("，", "。", "？", "！"):
            print(f"    phoneme[{i}] = '{phone_str}' (punctuation)")
            # find next silence
            found = False
            while tg_idx < len(timeline):
                tl = timeline[tg_idx]
                if tl["is_silence"]:
                    print(f"      -> silence at tg[{tg_idx}] dur={tl['dur_s']:.3f}s")
                    tg_idx += 1
                    found = True
                    break
                tg_idx += 1
            if not found:
                print(f"      -> NO SILENCE FOUND")
            continue
        
        # Find matching char
        char_found = False
        while tg_idx < len(timeline):
            tl = timeline[tg_idx]
            if tl["text"] == char and not tl["is_silence"]:
                print(f"    phoneme[{i}] = {phone_str} (char='{char}') -> matched tg[{tg_idx}] '{tl['text']}' dur={tl['dur_s']:.3f}s")
                tg_idx += 1
                char_found = True
                break
            tg_idx += 1
        
        if not char_found:
            print(f"    phoneme[{i}] = {phone_str} (char='{char}') -> NOT FOUND, default 0.05s")
    
    # Check the old durations for comparison
    print(f"  Old durations: {rec['durations'][:20]}")
    print(f"  Old sum={sum(rec['durations'])}")

# ============================================================================
# 3. Check: what's the TextGrid word tier actually aligned to?
# ============================================================================
print(f"\n{'='*70}")
print("3. TEXTGRID WORD TIER CHARACTER ANALYSIS")
print("="*70)

# The TextGrid 'words' tier should have individual Chinese characters
# Let's check what characters are in the TextGrid vs the poem text
mismatches = 0
total_checked = 0
for rec in manifest[:20]:
    pid = rec["poem_id"]
    text = rec["text"]
    tg_path = ROOT / "data" / "mfa_aligned" / f"{pid}.TextGrid"
    if not tg_path.exists():
        continue
    
    intervals = parse_textgrid_intervals(tg_path)
    tg_chars = [iv["text"] for iv in intervals if iv["text"] and not iv["is_silence"]]
    poem_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    
    if tg_chars != poem_chars:
        mismatches += 1
        print(f"\n  MISMATCH {pid}:")
        print(f"    TG:  {''.join(tg_chars[:20])}")
        print(f"    Text: {''.join(poem_chars[:20])}")
        # Show first diff
        for j in range(min(len(tg_chars), len(poem_chars))):
            if j >= len(tg_chars) or j >= len(poem_chars) or tg_chars[j] != poem_chars[j]:
                print(f"    First diff at [{j}]: TG='{tg_chars[j] if j < len(tg_chars) else 'END'}' vs poem='{poem_chars[j] if j < len(poem_chars) else 'END'}'")
                break
    total_checked += 1

print(f"\nMismatches: {mismatches}/{total_checked}")

# ============================================================================
# 4. Compute what the scaled durations SHOULD look like
# ============================================================================
print(f"\n{'='*70}")
print("4. SCALED DURATION ANALYSIS")
print("="*70)

# For each poem: compute the per-character duration in seconds from TextGrid,
# then scale to mel frames. What does the distribution look like BEFORE rounding?
all_raw_scaled = []
for rec in manifest[:50]:
    pid = rec["poem_id"]
    mel_len = rec["mel_len"]
    text = rec["text"]
    tg_path = ROOT / "data" / "mfa_aligned" / f"{pid}.TextGrid"
    if not tg_path.exists():
        continue
    
    intervals = parse_textgrid_intervals(tg_path)
    tg_total = intervals[-1]["xmax"] if intervals else 1.0
    
    # For each non-silence interval, compute scaled duration in mel frames
    char_durs_s = []
    for iv in intervals:
        if iv["text"] and not iv["is_silence"]:
            char_durs_s.append(iv["xmax"] - iv["xmin"])
    
    if sum(char_durs_s) == 0:
        continue
    
    scale = mel_len / (sum(char_durs_s) * FRAME_RATE)
    for d in char_durs_s:
        all_raw_scaled.append(d * scale * FRAME_RATE)

all_raw_scaled = np.array(all_raw_scaled)
print(f"Raw scaled durations (before rounding), {len(all_raw_scaled)} chars from 50 poems:")
print(f"  Mean: {np.mean(all_raw_scaled):.2f} frames")
print(f"  Median: {np.median(all_raw_scaled):.2f} frames")
print(f"  Std: {np.std(all_raw_scaled):.2f}")
print(f"  Range: [{np.min(all_raw_scaled):.2f}, {np.max(all_raw_scaled):.2f}]")
print(f"  Percentiles: 10%={np.percentile(all_raw_scaled, 10):.2f}  25%={np.percentile(all_raw_scaled, 25):.2f} "
      f"50%={np.percentile(all_raw_scaled, 50):.2f}  75%={np.percentile(all_raw_scaled, 75):.2f} "
      f"90%={np.percentile(all_raw_scaled, 90):.2f}")
print(f"\n  After rounding (max(round(x), 1)):")
rounded = np.maximum(np.round(all_raw_scaled), 1)
c = Counter(rounded.astype(int).tolist())
for k in sorted(c.keys())[:15]:
    pct = c[k] / len(rounded) * 100
    print(f"    dur={k:>3}: {c[k]:>5} ({pct:>5.1f}%)")

print("\nDONE")
