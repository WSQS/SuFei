"""Filter teacher audio by ASR quality. Outputs clean_data_list.json."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data" / "audio"
TRAIN_JSONL = ROOT / "data" / "train_raw.jsonl"
OUTPUT = ROOT / "data" / "clean_data_list.json"

PUNCT = re.compile(r"[，。！？；：、,.!?;:\"'""''（）()\[\]【】《》〈〉…—\-．\.\n\r\s\u3000]")

from opencc import OpenCC
cc = OpenCC("t2s")

def normalize(text):
    text = cc.convert(text)
    return PUNCT.sub("", text)

def extract_content(full_text):
    parts = full_text.split("。", 1)
    return parts[1] if len(parts) > 1 else full_text

def compute_cer_fast(ref, hyp):
    """Simple Levenshtein distance / len(ref)."""
    m, n = len(ref), len(hyp)
    if m == 0:
        return 1.0, 0, 0, 0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    # Count operations
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
    return (s + d + ins) / m, s, d, ins


def main():
    from funasr import AutoModel
    import torchaudio

    # Load training data
    records = []
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    print(f"Total records: {len(records)}")

    # Load ASR
    print("Loading ASR model...")
    asr = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device="cuda",
        disable_update=True,
    )
    print("ASR loaded.")

    results = []
    passed = 0
    failed = 0

    for i, r in enumerate(records):
        wav_name = Path(r["audio"]).name
        wav_path = AUDIO_DIR / wav_name
        if not wav_path.exists():
            print(f"  [{i+1}/{len(records)}] SKIP: {wav_name} not found")
            continue

        full_text = r["text"]
        content = extract_content(full_text)
        # Compare ASR against FULL text (title+author+content), not stripped
        ref_full_norm = normalize(full_text)

        # ASR transcribe
        result = asr.generate(input=str(wav_path), batch_size_s=300)
        asr_text = result[0].get("text", "") if result else ""
        asr_norm = normalize(asr_text)

        # Single CER: full text vs ASR (both include title/author)
        cer, s, d, ins = compute_cer_fast(ref_full_norm, asr_norm)

        # Coverage: how much of full text appears in ASR
        ref_len = len(ref_full_norm)
        coverage = sum(1 for ch in ref_full_norm if ch in asr_norm) / ref_len if ref_len else 0

        # Audio duration
        info = torchaudio.info(str(wav_path))
        duration = info.num_frames / info.sample_rate

        # Line completeness (rough check)
        ref_len = len(ref_full_norm)
        coverage = sum(1 for ch in ref_full_norm if ch in asr_norm) / ref_len if ref_len else 0

        # Quality gates (relaxed: ASR has inherent ~15% CER on teacher audio)
        pass_cer = cer < 0.50
        pass_coverage = coverage >= 0.70
        pass_duration = 3.0 < duration < 120.0
        is_clean = pass_cer and pass_coverage and pass_duration

        if is_clean:
            passed += 1
        else:
            failed += 1

        status = "OK" if is_clean else "DROP"
        reasons = []
        if not pass_cer:
            reasons.append(f"CER={cer:.1%}")
        if not pass_coverage:
            reasons.append(f"COV={coverage:.1%}")
        if not pass_duration:
            reasons.append(f"DUR={duration:.1f}s")

        title = full_text.split("，")[0]
        reason_str = f" ({'; '.join(reasons)})" if reasons else ""
        print(f"  [{i+1}/{len(records)}] {title:15s} CER={cer:.1%} COV={coverage:.0%} D={duration:.1f}s [{status}]{reason_str}")

        results.append({
            "index": i,
            "audio": str(wav_path),
            "title": title,
            "full_text": full_text,
            "content": content,
            "ref_normalized": ref_full_norm,
            "asr_normalized": asr_norm,
            "asr_raw": asr_text,
            "cer": round(cer, 4),
            "deletions": d,
            "insertions": ins,
            "substitutions": s,
            "coverage": round(coverage, 4),
            "duration": round(duration, 2),
            "clean": is_clean,
            "reasons": reasons,
        })

    # Save
    clean_records = [r for r in results if r["clean"]]
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(results),
            "clean": len(clean_records),
            "dropped": len(results) - len(clean_records),
            "samples": clean_records,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Filtering complete: {len(clean_records)}/{len(results)} clean ({len(clean_records)/len(results)*100:.0f}%)")
    print(f"Dropped: {len(results) - len(clean_records)}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
