"""Rebuild all manifests with fixed duration builder.

Fixes KNOWN_ISSUES #6: MFA TextGrid timescale mismatch + char matching.
The fixed builder scales TextGrid timestamps proportionally to mel_len,
and falls back to 50ms default for unmatched characters.

Rebuilds: paddle_distill_manifest, train_171, train_300, holdout_20,
external_unseen_6.
"""
import json, sys, shutil
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_paddlespeech_distillation_data import (
    parse_textgrid_intervals, text_to_phonemes, FRAME_RATE, INITIAL_RATIO,
)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "paddle_distill_features"
MFA_DIR = ROOT / "data" / "mfa_aligned"
MFA_NEW129_DIR = ROOT / "data" / "mfa_new129_aligned"


def build_durations_fixed(phonemes, tg_intervals, mel_len):
    """Build per-phoneme durations from TextGrid, proportionally scaled to mel_len."""
    if not tg_intervals:
        return [max(mel_len // len(phonemes), 1)] * len(phonemes)

    durations_s = []
    tg_idx = 0
    timeline = []
    for iv in tg_intervals:
        timeline.append({
            "text": iv["text"],
            "dur_s": iv["xmax"] - iv["xmin"],
            "is_silence": iv["is_silence"],
        })

    i = 0
    while i < len(phonemes):
        phone_str, char = phonemes[i]

        if phone_str == "<eos>":
            durations_s.append(0.0)
            i += 1
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
            durations_s.append(silence_dur_s)
            i += 1
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

        if i + 1 < len(phonemes) and phonemes[i + 1][1] == char:
            init_dur_s = char_dur_s * INITIAL_RATIO
            final_dur_s = char_dur_s - init_dur_s
            durations_s.append(init_dur_s)
            durations_s.append(final_dur_s)
            i += 2
        else:
            durations_s.append(char_dur_s)
            i += 1

    # Scale seconds to mel frames
    eos_mask = [p[0] == "<eos>" for p in phonemes]
    scale_targets = [d for d, m in zip(durations_s, eos_mask) if not m]
    total_s = sum(scale_targets)
    if total_s > 0:
        scale_factor = mel_len / (total_s * FRAME_RATE)
    else:
        scale_factor = 1.0

    result = []
    for d, is_eos in zip(durations_s, eos_mask):
        if is_eos:
            result.append(1)
        else:
            result.append(max(round(d * scale_factor * FRAME_RATE), 1))

    # Fix sum to match mel_len exactly
    diff = mel_len - sum(result)
    if diff != 0:
        max_idx = max(range(len(result)), key=lambda x: result[x])
        result[max_idx] = max(result[max_idx] + diff, 1)

    return result


def find_textgrid(pid):
    """Find TextGrid in either MFA output directory."""
    for d in [MFA_DIR, MFA_NEW129_DIR]:
        p = d / f"{pid}.TextGrid"
        if p.exists():
            return p
    return None


def rebuild_manifest(manifest_path, output_path):
    """Rebuild durations for all records in a manifest."""
    records = [json.loads(l) for l in open(manifest_path, encoding="utf-8")]
    fixed = []
    stats = {"total": 0, "fixed": 0, "no_tg": 0, "sum_mismatch": 0}

    for rec in records:
        stats["total"] += 1
        pid = rec["poem_id"]
        mel_len = rec["mel_len"]
        text = rec["text"]

        tg_path = find_textgrid(pid)
        if tg_path is None:
            print(f"  WARN {pid}: no TextGrid found, keeping old durations")
            fixed.append(rec)
            stats["no_tg"] += 1
            continue

        intervals = parse_textgrid_intervals(tg_path)
        phonemes = text_to_phonemes(text)
        durations = build_durations_fixed(phonemes, intervals, mel_len)

        # Force exact match by distributing rounding errors
        diff = mel_len - sum(durations)
        if diff > 0:
            # Add 1 frame to the largest `diff` durations
            sorted_idx = sorted(range(len(durations)), key=lambda x: -durations[x])
            for j in range(diff):
                durations[sorted_idx[j % len(durations)]] += 1
        elif diff < 0:
            # Remove 1 frame from the largest durations
            sorted_idx = sorted(range(len(durations)), key=lambda x: -durations[x])
            for j in range(-diff):
                idx = sorted_idx[j]
                if durations[idx] > 1:
                    durations[idx] -= 1
        assert sum(durations) == mel_len, f"{pid}: final sum={sum(durations)} != mel_len={mel_len}"

        rec["durations"] = durations
        fixed.append(rec)
        stats["fixed"] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for r in fixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return fixed, stats


def main():
    # Backup originals
    backup_dir = ROOT / "data" / "manifest_backups_pre_durfix"
    backup_dir.mkdir(exist_ok=True)

    manifests = [
        ("paddle_distill_manifest.jsonl", None),
        ("train_171_manifest.jsonl", None),
        ("train_300_manifest.jsonl", None),
        ("holdout_20_manifest.jsonl", None),
        ("external_unseen_6_manifest.jsonl", None),
    ]

    all_results = []
    for fname, _ in manifests:
        path = ROOT / "data" / fname
        if not path.exists():
            print(f"SKIP {fname}: not found")
            continue

        # Backup
        backup_path = backup_dir / fname
        shutil.copy2(path, backup_path)

        print(f"\n{'='*60}")
        print(f"Rebuilding {fname}")
        print(f"{'='*60}")

        fixed, stats = rebuild_manifest(path, path)
        all_results.append((fname, fixed, stats))

        print(f"  Total: {stats['total']}, Fixed: {stats['fixed']}, "
              f"No TG: {stats['no_tg']}, Mismatch: {stats['sum_mismatch']}")

    # Summary statistics across train_300
    print(f"\n{'='*60}")
    print("Duration statistics after fix (train_300)")
    print(f"{'='*60}")

    train300 = [json.loads(l) for l in open(ROOT / "data" / "train_300_manifest.jsonl", encoding="utf-8")]
    all_first = []
    all_nonfirst = []
    for rec in train300:
        d = rec["durations"]
        all_first.append(d[0])
        all_nonfirst.extend(d[1:])

    all_first = np.array(all_first)
    all_nonfirst = np.array(all_nonfirst)

    print(f"First phoneme: mean={all_first.mean():.1f} median={np.median(all_first):.0f} "
          f"min={all_first.min()} max={all_first.max()}")
    print(f"Non-first: mean={all_nonfirst.mean():.1f} median={np.median(all_nonfirst):.0f} "
          f"min={all_nonfirst.min()} max={all_nonfirst.max()}")
    cnt = Counter(all_nonfirst.tolist())
    print(f"Non-first =2: {cnt.get(2, 0)}/{len(all_nonfirst)} "
          f"({cnt.get(2, 0)/len(all_nonfirst)*100:.1f}%)")

    # Verify invariants
    bad = sum(1 for r in train300 if sum(r["durations"]) != r["mel_len"])
    print(f"Duration invariant violations: {bad}/{len(train300)}")

    # Verify train_171 ⊂ train_300
    t171 = [json.loads(l) for l in open(ROOT / "data" / "train_171_manifest.jsonl", encoding="utf-8")]
    t171_ids = {r["poem_id"] for r in t171}
    t300_ids = {r["poem_id"] for r in train300}
    print(f"train_171 subset of train_300: {t171_ids.issubset(t300_ids)}")

    print("\nDone. Backups in:", backup_dir)


if __name__ == "__main__":
    main()
