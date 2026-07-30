"""Deep analysis of 217 clean samples: pinyin error rate, D/S/I breakdown, audit selection."""
import json
import re
import statistics
from pathlib import Path
from collections import Counter

from opencc import OpenCC
from pypinyin import pinyin, Style

ROOT = Path(__file__).resolve().parents[1]
cc = OpenCC("t2s")

PUNCT = re.compile(r"[，。！？；：、,.!?;:\"'""''（）()\[\]【】《》〈〉…—\-．\.\n\r\s\u3000]")

def normalize(text):
    return PUNCT.sub("", cc.convert(text))

def to_pinyin(text):
    """Convert normalized text to pinyin syllables (no tones)."""
    syllables = []
    for char in text:
        py = pinyin(char, style=Style.NORMAL, errors="ignore")
        if py and py[0] and py[0][0] != char:
            syllables.append(py[0][0])
    return syllables

def levenshtein(ref, hyp):
    m, n = len(ref), len(hyp)
    if m == 0:
        return 0, 0, n  # all insertions
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    i, j = m, n
    s = d = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1; j -= 1
        elif i > 0 and (j == 0 or dp[i-1][j] <= dp[i][j-1]):
            d += 1; i -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] < dp[i-1][j-1]):
            ins += 1; j -= 1
        else:
            s += 1; i -= 1; j -= 1
    return s, d, ins

def main():
    with open(ROOT / "data" / "clean_data_list.json", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    print(f"Analyzing {len(samples)} clean samples\n")

    # Compute pinyin error rates
    results = []
    for s in samples:
        ref_text = s["ref_normalized"]
        asr_text = s["asr_normalized"]

        # Hanzi metrics
        h_s, h_d, h_ins = levenshtein(ref_text, asr_text)
        h_n = len(ref_text)
        h_cer = (h_s + h_d + h_ins) / h_n if h_n else 1.0
        h_del_rate = h_d / h_n if h_n else 0
        h_ins_rate = h_ins / h_n if h_n else 0
        h_sub_rate = h_s / h_n if h_n else 0

        # Pinyin metrics
        ref_py = to_pinyin(ref_text)
        asr_py = to_pinyin(asr_text)
        p_s, p_d, p_ins = levenshtein(ref_py, asr_py)
        p_n = len(ref_py)
        p_per = (p_s + p_d + p_ins) / p_n if p_n else 1.0
        p_del_rate = p_d / p_n if p_n else 0
        p_ins_rate = p_ins / p_n if p_n else 0

        # Trailing deletion (hanzi)
        trailing = 0
        ops = []
        # Recompute alignment for trailing
        # Simple: count consecutive missing chars from end of ref
        # Check last chars of ref not in asr
        ref_rev = ref_text[::-1]
        asr_set = set(asr_text)
        for ch in ref_rev:
            if ch not in asr_text[-len(ref_text):]:
                trailing += 1
            else:
                break
        trailing = min(trailing, h_d)  # can't exceed total deletions

        results.append({
            **s,
            "hanzi_cer": h_cer,
            "hanzi_del_rate": h_del_rate,
            "hanzi_ins_rate": h_ins_rate,
            "hanzi_sub_rate": h_sub_rate,
            "pinyin_per": p_per,
            "pinyin_del_rate": p_del_rate,
            "pinyin_ins_rate": p_ins_rate,
            "trailing_del": trailing,
            "trailing_del_ratio": trailing / h_n if h_n else 0,
        })

    # Summary stats
    hanzi_cers = [r["hanzi_cer"] for r in results]
    pinyin_pers = [r["pinyin_per"] for r in results]
    hanzi_dels = [r["hanzi_del_rate"] for r in results]
    hanzi_inss = [r["hanzi_ins_rate"] for r in results]
    hanzi_subs = [r["hanzi_sub_rate"] for r in results]

    print("=" * 70)
    print("Summary Statistics (217 clean samples)")
    print("=" * 70)
    print(f"  Hanzi CER:     median={statistics.median(hanzi_cers):.1%}, mean={statistics.mean(hanzi_cers):.1%}")
    print(f"  Pinyin PER:    median={statistics.median(pinyin_pers):.1%}, mean={statistics.mean(pinyin_pers):.1%}")
    print(f"  Hanzi Del:     median={statistics.median(hanzi_dels):.1%}, mean={statistics.mean(hanzi_dels):.1%}")
    print(f"  Hanzi Ins:     median={statistics.median(hanzi_inss):.1%}, mean={statistics.mean(hanzi_inss):.1%}")
    print(f"  Hanzi Sub:     median={statistics.median(hanzi_subs):.1%}, mean={statistics.mean(hanzi_subs):.1%}")

    # Interpretation
    print(f"\n{'='*70}")
    print("Error Type Analysis")
    print(f"{'='*70}")
    sub_dominant = sum(1 for r in results if r["hanzi_sub_rate"] > r["hanzi_del_rate"] + r["hanzi_ins_rate"])
    del_dominant = sum(1 for r in results if r["hanzi_del_rate"] > r["hanzi_sub_rate"])
    ins_dominant = sum(1 for r in results if r["hanzi_ins_rate"] > r["hanzi_sub_rate"])
    print(f"  Substitution-dominant (likely homophones): {sub_dominant}/{len(results)} ({sub_dominant/len(results)*100:.0f}%)")
    print(f"  Deletion-dominant (likely missing chars):  {del_dominant}/{len(results)} ({del_dominant/len(results)*100:.0f}%)")
    print(f"  Insertion-dominant (likely extra content): {ins_dominant}/{len(results)} ({ins_dominant/len(results)*100:.0f}%)")

    # Pinyin vs Hanzi gap
    py_better = sum(1 for r in results if r["pinyin_per"] < r["hanzi_cer"] - 0.05)
    print(f"\n  Pinyin PER significantly lower than Hanzi CER (>5% gap): {py_better}/{len(results)} ({py_better/len(results)*100:.0f}%)")
    print(f"  → These are likely homophone substitution errors (audio is correct)")

    # Total audio duration
    total_dur = sum(r["duration"] for r in results)
    print(f"\n  Total audio: {total_dur:.0f}s ({total_dur/60:.1f} min)")

    # Select 30 audit samples
    print(f"\n{'='*70}")
    print("Audit Selection (30 samples)")
    print(f"{'='*70}")

    sorted_by_cer = sorted(results, key=lambda r: r["hanzi_cer"])

    # Low CER (best quality)
    low = sorted_by_cer[:10]
    # Median CER
    mid_start = len(sorted_by_cer) // 2 - 5
    mid = sorted_by_cer[mid_start:mid_start+10]
    # High CER (still PASS)
    high = sorted_by_cer[-10:]

    print("\n--- Low CER (best 10) ---")
    for r in low:
        print(f"  {r['title']:20s} CER={r['hanzi_cer']:.1%} PER={r['pinyin_per']:.1%} D={r['hanzi_del_rate']:.1%} I={r['hanzi_ins_rate']:.1%} S={r['hanzi_sub_rate']:.1%} | {r['content'][:30]}")

    print("\n--- Median CER (10) ---")
    for r in mid:
        print(f"  {r['title']:20s} CER={r['hanzi_cer']:.1%} PER={r['pinyin_per']:.1%} D={r['hanzi_del_rate']:.1%} I={r['hanzi_ins_rate']:.1%} S={r['hanzi_sub_rate']:.1%} | {r['content'][:30]}")

    print("\n--- High CER (worst 10, still PASS) ---")
    for r in high:
        print(f"  {r['title']:20s} CER={r['hanzi_cer']:.1%} PER={r['pinyin_per']:.1%} D={r['hanzi_del_rate']:.1%} I={r['hanzi_ins_rate']:.1%} S={r['hanzi_sub_rate']:.1%} | {r['content'][:30]}")

    # Save updated list
    output = ROOT / "data" / "clean_data_analysis.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(results),
            "total_duration_sec": round(total_dur, 1),
            "stats": {
                "hanzi_cer_median": round(statistics.median(hanzi_cers), 4),
                "hanzi_cer_mean": round(statistics.mean(hanzi_cers), 4),
                "pinyin_per_median": round(statistics.median(pinyin_pers), 4),
                "pinyin_per_mean": round(statistics.mean(pinyin_pers), 4),
                "sub_dominant_count": sub_dominant,
                "del_dominant_count": del_dominant,
                "ins_dominant_count": ins_dominant,
            },
            "audit_indices": {
                "low": [r["index"] for r in low],
                "median": [r["index"] for r in mid],
                "high": [r["index"] for r in high],
            },
            "samples": [{k: v for k, v in r.items() if k != "asr_raw"} for r in results],
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
