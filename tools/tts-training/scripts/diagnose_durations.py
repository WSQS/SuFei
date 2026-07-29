"""Diagnose duration distribution and mel silence ratio across the dataset."""
import json, sys, numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "paddle_distill_features"

for manifest_name, label in [
    ("train_300_manifest.jsonl", "TRAIN_300"),
    ("holdout_20_manifest.jsonl", "HOLDOUT_20"),
]:
    manifest = [json.loads(l) for l in open(ROOT / "data" / manifest_name, encoding="utf-8")]
    print(f"\n{'='*70}")
    print(f"  Duration analysis: {label} ({len(manifest)} poems)")
    print(f"{'='*70}")

    all_durs = []
    first_durs = []
    non_first_durs = []
    silence_ratios = []

    for rec in manifest:
        durs = rec["durations"]
        all_durs.extend(durs)
        first_durs.append(durs[0])
        non_first_durs.extend(durs[1:])

        npz = np.load(str(FEAT_DIR / f"{rec['poem_id']}.npz"))
        mel = npz["mel"]
        frame_energy = mel.mean(axis=1)
        silence_ratio = (frame_energy < -5).mean()
        silence_ratios.append(silence_ratio)

    all_durs = np.array(all_durs)
    first_durs = np.array(first_durs)
    non_first = np.array(non_first_durs)
    silence_ratios = np.array(silence_ratios)

    print(f"\n  First phoneme duration:")
    print(f"    mean={first_durs.mean():.1f} median={np.median(first_durs):.0f} "
          f"min={first_durs.min()} max={first_durs.max()}")
    print(f"    >100: {np.sum(first_durs > 100)}/{len(first_durs)} "
          f"({np.sum(first_durs > 100)/len(first_durs)*100:.0f}%)")
    print(f"    >50:  {np.sum(first_durs > 50)}/{len(first_durs)}")

    # What fraction of total duration does first phoneme take?
    for rec in manifest[:5]:
        total = sum(rec["durations"])
        first = rec["durations"][0]
        print(f"    {rec['poem_id']}: first={first}/{total} ({first/total*100:.0f}%) "
              f"text={rec['text'][:30]}")

    print(f"\n  Non-first phoneme duration:")
    print(f"    mean={non_first.mean():.2f} median={np.median(non_first):.0f} "
          f"min={non_first.min()} max={non_first.max()}")
    dur_counts = Counter(non_first.tolist())
    print(f"    Distribution: {dict(sorted(dur_counts.items())[:10])}")

    print(f"\n  All phoneme durations:")
    print(f"    mean={all_durs.mean():.1f} median={np.median(all_durs):.0f} "
          f"min={all_durs.min()} max={all_durs.max()}")
    print(f"    Histogram: {np.histogram(all_durs, bins=[0,1,2,3,5,10,50,100,500,1000])[0].tolist()}")

    print(f"\n  Mel silence ratio (frame energy < -5):")
    print(f"    mean={silence_ratios.mean():.1%} median={np.median(silence_ratios):.1%} "
          f"min={silence_ratios.min():.1%} max={silence_ratios.max():.1%}")
    print(f"    >50% silent: {np.sum(silence_ratios > 0.5)}/{len(silence_ratios)}")
