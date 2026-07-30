"""PaddleSpeech FS2 baseline on 24-poem test set.

Uses pre-trained PaddleSpeech FS2 ONNX + HiFiGAN vocoder.
No training — pure pre-trained model inference.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_stage1_e2e import text_to_phones, phones_to_ids, load_phone_id_map
from eval_tts import load_asr_model, transcribe, normalize_text, align

FS2_DIR = ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0"
HIFIGAN_DIR = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
TEST_SET = ROOT / "data" / "test_set_v1.json"
OUT = ROOT / "output" / "paddlespeech_baseline"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    phone_map = load_phone_id_map()
    fs2 = ort.InferenceSession(str(FS2_DIR / "fastspeech2_csmsc.onnx"))
    hifigan = ort.InferenceSession(str(HIFIGAN_DIR / "hifigan_csmsc.onnx"))

    with open(TEST_SET, encoding="utf-8") as f:
        test_data = json.load(f)
    samples = test_data["samples"]

    print("Loading ASR...")
    asr = load_asr_model()

    results = []
    total_infer = 0
    total_audio = 0

    for item in samples:
        poem_id = item["poem_id"]
        content = item["content"]
        ref = normalize_text(content)

        phones = text_to_phones(content)
        ids = phones_to_ids(phones, phone_map)

        if not ids:
            print(f"  SKIP {poem_id}: no valid phones")
            continue

        text_arr = np.array(ids, dtype=np.int64)
        t0 = time.time()
        mel = fs2.run(None, {"text": text_arr})[0]
        infer_time = time.time() - t0

        audio = hifigan.run(None, {"logmel": mel.astype(np.float32)})[0].flatten()
        audio_dur = len(audio) / 24000

        total_infer += infer_time
        total_audio += audio_dur

        wav_path = str(OUT / f"{poem_id}.wav")
        sf.write(wav_path, audio, 24000)

        asr_text, _ = transcribe(asr, wav_path)
        hyp = normalize_text(asr_text)
        ops = align(ref, hyp)
        d = sum(1 for r, h in ops if r != '*' and h == '*')
        s = sum(1 for r, h in ops if r != '*' and h != '*' and r != h)
        i = sum(1 for r, h in ops if r == '*' and h != '*')
        cer = (d + s) / max(len(ref), 1)

        status = "OK" if cer < 0.15 else ("MARGINAL" if cer < 0.3 else "FAIL")
        rtf = infer_time / audio_dur if audio_dur > 0 else 0

        print(f"  {poem_id}: CER={cer:.0%} {status} rtf={rtf:.3f} dur={audio_dur:.1f}s | {content[:25]}")

        results.append({"poem_id": poem_id, "cer": cer, "status": status,
                        "d": d, "s": s, "i": i, "rtf": rtf, "audio_dur": audio_dur})

    # Summary
    print(f"\n{'='*60}")
    print(f"PaddleSpeech FS2 Baseline ({len(results)} samples)")
    print(f"{'='*60}")

    cers = [r["cer"] for r in results]
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_marginal = sum(1 for r in results if r["status"] == "MARGINAL")

    print(f"  Mean CER:   {np.mean(cers):.1%}")
    print(f"  Median CER: {np.median(cers):.1%}")
    print(f"  P90 CER:    {np.percentile(cers, 90):.1%}")
    print(f"  OK (<15%):  {n_ok}/{len(results)}")
    print(f"  Marginal:   {n_marginal}/{len(results)}")
    print(f"  Avg RTF:    {total_infer/total_audio:.3f}")
    print(f"  Total audio: {total_audio:.0f}s")


if __name__ == "__main__":
    main()
