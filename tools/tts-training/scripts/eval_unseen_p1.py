"""Complete P1 (teacher wav -> ASR) for the 6 unseen poems.

Teacher wavs already exist at output/unseen_teacher_wav/.
P2/P3 results already in E3_unseen_novad/report.json.
This script only runs P1 and produces the full P1/P2/P3 table.
"""
import json, sys, numpy as np
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
sys.path.insert(0, str(ROOT / "scripts"))
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

# Load existing P2/P3 results
with open(ROOT / "output/unified_eval/E3_unseen_novad/report.json", encoding="utf-8") as f:
    existing = json.load(f)

# Load unseen manifest for texts
manifests = {}
with open(ROOT / "data/unseen_manifest.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        manifests[r["poem_id"]] = r

# Load ASR (with VAD+punc, same as eval_unified)
print("Loading ASR...")
asr = load_asr_model()
print("ASR loaded.")

teacher_dir = ROOT / "output" / "unseen_teacher_wav"

results = []
for sample in existing["samples"]:
    pid = sample["poem_id"]
    full_text = sample["full_text"]
    mel_len = sample["mel_len"]
    
    # P1: teacher wav -> ASR
    teacher_wav = str(teacher_dir / f"{pid}.wav")
    asr_text_1, dur1 = transcribe(asr, teacher_wav)
    cer1, _, _, _, _ = cer_detail(full_text, asr_text_1)
    
    # P2/P3 from existing
    cer2 = sample["cer2"]
    cer3 = sample["cer3"]
    delta = sample["student_delta"]
    l1 = sample["mel_l1"]
    
    recon_delta = cer2 - cer1
    # Classify by length
    is_short = mel_len < 800
    
    print(f"\n{pid} (mel_len={mel_len}, {'short' if is_short else 'LONG'}):")
    print(f"  P1={cer1:.0%}  P2={cer2:.0%}  P3={cer3:.0%}")
    print(f"  recon_delta={recon_delta:+.0%}  student_delta={delta:+.0%}")
    print(f"  mel_L1={l1:.3f}")
    print(f"  P1 asr: {asr_text_1[:60]}")
    print(f"  P2 asr: {sample['asr2'][:60]}")
    print(f"  P3 asr: {sample['asr3'][:60]}")
    
    results.append({
        "poem_id": pid,
        "mel_len": mel_len,
        "is_short": is_short,
        "mel_l1": l1,
        "cer1": cer1,
        "cer2": cer2,
        "cer3": cer3,
        "recon_delta": recon_delta,
        "student_delta": delta,
        "full_text": full_text,
        "asr1": asr_text_1,
        "asr2": sample["asr2"],
        "asr3": sample["asr3"],
    })

# Summary
print(f"\n{'='*70}")
print(f"Full P1/P2/P3 Summary (n={len(results)})")
print(f"{'='*70}")
print(f"{'Poem':<30} {'mel_len':>7} {'L1':>5} {'P1':>5} {'P2':>5} {'P3':>5} {'recon':>6} {'student':>8}")
print("-" * 70)
for r in results:
    name = r["poem_id"].replace("unseen_", "")
    print(f"{name:<30} {r['mel_len']:>7} {r['mel_l1']:>.3f} {r['cer1']:>4.0%} {r['cer2']:>4.0%} {r['cer3']:>4.0%} {r['recon_delta']:>+5.0%} {r['student_delta']:>+7.0%}")

# Group stats
short = [r for r in results if r["is_short"]]
long_ = [r for r in results if not r["is_short"]]
print(f"\nShort poems (mel_len < 800): n={len(short)}")
if short:
    print(f"  P1 mean={np.mean([r['cer1'] for r in short]):.0%}  P2 mean={np.mean([r['cer2'] for r in short]):.0%}  P3 mean={np.mean([r['cer3'] for r in short]):.0%}")
    print(f"  student_delta mean={np.mean([r['student_delta'] for r in short]):+.0%}")
    print(f"  mel L1 mean={np.mean([r['mel_l1'] for r in short]):.3f}")
print(f"Long poems (mel_len >= 800): n={len(long_)}")
if long_:
    print(f"  P1 mean={np.mean([r['cer1'] for r in long_]):.0%}  P2 mean={np.mean([r['cer2'] for r in long_]):.0%}  P3 mean={np.mean([r['cer3'] for r in long_]):.0%}")
    print(f"  student_delta mean={np.mean([r['student_delta'] for r in long_]):+.0%}")
    print(f"  mel L1 mean={np.mean([r['mel_l1'] for r in long_]):.3f}")

# Save
out = ROOT / "output" / "unified_eval" / "E3_unseen_full"
out.mkdir(parents=True, exist_ok=True)
report = {
    "name": "E3_unseen_full_p1p2p3",
    "n_samples": len(results),
    "samples": results,
}
with open(out / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\nReport saved to {out / 'report.json'}")
