"""V1/V2/V3 GT mel vocoder test (no model involved).

Strict controlled experiment:
- V1: mel_raw → HiFiGAN → ASR CER
- V2: mel_norm → HiFiGAN → ASR CER
- V3: mel_roundtrip = mel_norm * std + mean → HiFiGAN → ASR CER

Tests on BOTH A features (nar_features) and B' features (paddle_distill_features)
using their respective norm_stats.

Also prints max|mel_raw - mel_roundtrip| to verify numerical equivalence.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_tts import load_asr_model, transcribe, normalize_text, align

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
OUT = ROOT / "output" / "v123_test"


def vocode(mel, sess):
    return sess.run(None, {"logmel": mel.astype(np.float32)})[0].flatten()


def cer_eval(asr, wav_path, ref_text):
    asr_text, _ = transcribe(asr, wav_path)
    hyp = normalize_text(asr_text)
    ref = normalize_text(ref_text)
    ops = align(ref, hyp)
    subs_d = sum(1 for r, h in ops if r != '*' and h == '*')
    subs_s = sum(1 for r, h in ops if r != '*' and h != '*' and r != h)
    return (subs_d + subs_s) / max(len(ref), 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    voc_sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))

    print("Loading ASR...")
    asr = load_asr_model()

    configs = [
        {
            "name": "A (CosyVoice)",
            "manifest": "train_manifest.jsonl",
            "features_dir": "nar_features",
            "stats_file": "norm_stats.json",
        },
        {
            "name": "B' (PaddleSpeech)",
            "manifest": "paddle_distill_manifest.jsonl",
            "features_dir": "paddle_distill_features",
            "stats_file": "paddle_distill_norm_stats.json",
        },
    ]

    targets = ["poem_0285", "poem_0244", "poem_0097"]

    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"Dataset: {cfg['name']}")
        print(f"{'='*70}")

        with open(ROOT / "data" / cfg["stats_file"]) as f:
            stats = json.load(f)
        mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
        mel_std = np.array(stats["mel_std"], dtype=np.float32)

        manifest = [json.loads(l) for l in open(ROOT / "data" / cfg["manifest"], encoding="utf-8")]
        manifest.sort(key=lambda r: r["mel_len"])

        for target_id in targets:
            rec = next((r for r in manifest if r["poem_id"] == target_id), None)
            if rec is None:
                print(f"  {target_id}: not in {cfg['name']} dataset, skipping")
                continue

            npz = np.load(str(ROOT / "data" / cfg["features_dir"] / f"{target_id}.npz"))
            mel_raw = npz["mel"].astype(np.float32)
            mel_norm = (mel_raw - mel_mean) / mel_std
            mel_roundtrip = mel_norm * mel_std + mel_mean

            full_text = rec["text"]
            content = full_text.split("。", 1)[1] if "。" in full_text else full_text

            # Verify roundtrip
            rt_diff = np.abs(mel_raw - mel_roundtrip)

            print(f"\n  {target_id}: {content[:40]}")
            print(f"    mel_raw:       mean={mel_raw.mean():.3f}, std={mel_raw.std():.3f}, range=[{mel_raw.min():.2f}, {mel_raw.max():.2f}]")
            print(f"    mel_norm:      mean={mel_norm.mean():.3f}, std={mel_norm.std():.3f}, range=[{mel_norm.min():.2f}, {mel_norm.max():.2f}]")
            print(f"    mel_roundtrip: mean={mel_roundtrip.mean():.3f}, std={mel_roundtrip.std():.3f}, range=[{mel_roundtrip.min():.2f}, {mel_roundtrip.max():.2f}]")
            print(f"    |raw - roundtrip|: max={rt_diff.max():.6f}, mean={rt_diff.mean():.6f}")

            # V1/V2/V3
            for vname, mel_in in [("V1_raw", mel_raw), ("V2_norm", mel_norm), ("V3_roundtrip", mel_roundtrip)]:
                audio = vocode(mel_in, voc_sess)
                wav_path = str(OUT / f"{cfg['name']}_{target_id}_{vname}.wav")
                sf.write(wav_path, audio, 24000)
                cer = cer_eval(asr, wav_path, content)
                status = "OK" if cer < 0.15 else ("MARG" if cer < 0.3 else "FAIL")
                print(f"    {vname:14s}: CER={cer:.0%} {status}")


if __name__ == "__main__":
    main()
