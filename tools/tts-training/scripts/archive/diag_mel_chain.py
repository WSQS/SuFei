"""Diagnostic: mel → vocoder → audio chain analysis.

Hypothesis: model outputs mel in NORMALIZED space (mean/std removed),
but HiFiGAN expects raw log-mel. Missing denormalization step.

Tests:
1. GT mel (raw) → vocoder → ASR CER  (should be good - baseline)
2. GT mel (normalized) → vocoder → ASR CER  (probably bad)
3. GT mel (normalized → denormalized) → vocoder → ASR CER  (should match test 1)
4. Model output mel → vocoder → ASR CER  (current A0 path)
5. Model output mel (denormalized) → vocoder → ASR CER  (proposed fix)
6. Compare mel statistics: GT raw vs GT normalized vs model output
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe, normalize_text, align

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
OUT = ROOT / "output" / "mel_chain_diag"

device = "cpu"


def vocode(mel, sess):
    return sess.run(None, {"logmel": mel.astype(np.float32)})[0].flatten()


def run_model_a0(model, phone_ids, dur_gt_t, pitch_gt, energy_gt_t):
    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        mel_input = model.length_regulator(x, dur_gt_t)
        T_out = mel_input.size(1)
        mel_input = mel_input + \
            model.pitch_embed(pitch_gt[:, :T_out].unsqueeze(-1)) + \
            model.energy_embed(energy_gt_t[:, :T_out].unsqueeze(-1))
        mel_input = model.pos_enc(mel_input)
        dec = mel_input
        for layer in model.decoder_layers:
            dec = layer(dec)
        mel = model.mel_linear(dec)
    return mel[0].numpy()


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

    with open(ROOT / "data" / "paddle_distill_norm_stats.json") as f:
        stats = json.load(f)
    mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
    mel_std = np.array(stats["mel_std"], dtype=np.float32)

    voc_sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))

    print("Loading ASR...")
    asr = load_asr_model()

    # Load model checkpoint
    ckpt = torch.load(str(ROOT / "checkpoints" / "fs2_bprime_8overfit.pt"),
                      map_location=device, weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Test on poem_0244 (静夜思, known CER=0%) and poem_0097 (竹里馆, known CER=55%)
    manifest = [json.loads(l) for l in open(ROOT / "data" / "paddle_distill_manifest.jsonl", encoding="utf-8")]
    manifest.sort(key=lambda r: r["mel_len"])

    for target_id in ["poem_0244", "poem_0097"]:
        rec = next(r for r in manifest if r["poem_id"] == target_id)
        npz = np.load(str(ROOT / "data" / "paddle_distill_features" / f"{target_id}.npz"))

        mel_gt_raw = npz["mel"].astype(np.float32)      # Raw log-mel from feature extraction
        mel_gt_norm = (mel_gt_raw - mel_mean) / mel_std  # Normalized (what model trains on)
        f0_gt = npz["f0"].astype(np.float32)
        e_gt = npz["energy"].astype(np.float32)
        dur_gt = np.array(rec["durations"], dtype=np.int32)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long, device=device)
        dur_gt_t = torch.tensor(np.array([dur_gt]), dtype=torch.long, device=device)
        f0_norm = np.where(f0_gt > 0, (np.log(np.maximum(f0_gt, 1)) - stats["f0_mean"]) / stats["f0_std"], 0.0).astype(np.float32)
        e_norm = ((e_gt - stats["energy_mean"]) / stats["energy_std"]).astype(np.float32)
        pitch_gt = torch.tensor(f0_norm, device=device).unsqueeze(0)
        energy_gt_t = torch.tensor(e_norm, device=device).unsqueeze(0)

        full_text = rec["text"]
        content = full_text.split("。", 1)[1] if "。" in full_text else full_text

        # Model output (normalized space)
        mel_model = run_model_a0(model, phone_ids, dur_gt_t, pitch_gt, energy_gt_t)

        # Denormalize model output
        mel_model_denorm = mel_model * mel_std + mel_mean

        print(f"\n{'='*70}")
        print(f"{target_id}: {content[:40]}")
        print(f"{'='*70}")

        # ── Mel statistics ──
        print(f"\n  Mel Statistics:")
        print(f"    GT raw:      mean={mel_gt_raw.mean():.3f}, std={mel_gt_raw.std():.3f}, min={mel_gt_raw.min():.3f}, max={mel_gt_raw.max():.3f}")
        print(f"    GT norm:     mean={mel_gt_norm.mean():.3f}, std={mel_gt_norm.std():.3f}, min={mel_gt_norm.min():.3f}, max={mel_gt_norm.max():.3f}")
        print(f"    Model out:   mean={mel_model.mean():.3f}, std={mel_model.std():.3f}, min={mel_model.min():.3f}, max={mel_model.max():.3f}")
        print(f"    Model denorm:mean={mel_model_denorm.mean():.3f}, std={mel_model_denorm.std():.3f}, min={mel_model_denorm.min():.3f}, max={mel_model_denorm.max():.3f}")

        # ── Vocoder tests ──
        tests = [
            ("1_GT_raw", mel_gt_raw),
            ("2_GT_norm", mel_gt_norm),
            ("3_GT_renorm", mel_gt_norm * mel_std + mel_mean),
            ("4_model_norm", mel_model),
            ("5_model_denorm", mel_model_denorm),
        ]

        print(f"\n  Vocoder Tests:")
        for name, mel_in in tests:
            T = mel_in.shape[0]
            mel_2d = mel_in  # HiFiGAN expects [T, n_mels] (2D)
            audio = vocode(mel_2d, voc_sess)
            wav_path = str(OUT / f"{target_id}_{name}.wav")
            sf.write(wav_path, audio, 24000)
            cer = cer_eval(asr, wav_path, content)
            dur = len(audio) / 24000
            status = "OK" if cer < 0.15 else ("MARG" if cer < 0.3 else "FAIL")
            print(f"    {name:20s}: CER={cer:.0%} dur={dur:.1f}s {status}")

    # Also check HiFiGAN input shape expectation
    print(f"\n{'='*70}")
    print(f"HiFiGAN ONNX Input/Output Shapes")
    print(f"{'='*70}")
    for inp in voc_sess.get_inputs():
        print(f"  Input:  name={inp.name}, shape={inp.shape}, type={inp.type}")
    for out in voc_sess.get_outputs():
        print(f"  Output: name={out.name}, shape={out.shape}, type={out.type}")


if __name__ == "__main__":
    main()
