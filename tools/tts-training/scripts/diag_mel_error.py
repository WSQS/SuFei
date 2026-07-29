"""Analyze where the mel L1 error is concentrated.

mel L1 = 0.029 looks small, but if it's concentrated on formant frequencies
that ASR depends on, the audio could be unintelligible.

Also compares: model output vs GT, both through vocoder, to isolate
"model error" from "vocoder error".
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
OUT = ROOT / "output" / "mel_error_analysis"

device = "cpu"


def vocode(mel, sess):
    return sess.run(None, {"logmel": mel.astype(np.float32)})[0].flatten()


def run_a0(model, phone_ids, dur_gt_t, pitch_gt, energy_gt_t):
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

    ckpt = torch.load(str(ROOT / "checkpoints" / "fs2_bprime_8overfit.pt"),
                      map_location=device, weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    manifest = [json.loads(l) for l in open(ROOT / "data" / "paddle_distill_manifest.jsonl", encoding="utf-8")]
    manifest.sort(key=lambda r: r["mel_len"])

    for target_id in ["poem_0244", "poem_0097", "poem_0298"]:
        rec = next(r for r in manifest if r["poem_id"] == target_id)
        npz = np.load(str(ROOT / "data" / "paddle_distill_features" / f"{target_id}.npz"))

        mel_gt_raw = npz["mel"].astype(np.float32)
        mel_gt_norm = (mel_gt_raw - mel_mean) / mel_std
        f0_gt = npz["f0"].astype(np.float32)
        e_gt = npz["energy"].astype(np.float32)
        dur_gt = np.array(rec["durations"], dtype=np.int32)
        phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long, device=device)
        dur_gt_t = torch.tensor(np.array([dur_gt]), dtype=torch.long, device=device)
        f0_norm = np.where(f0_gt > 0, (np.log(np.maximum(f0_gt, 1)) - stats["f0_mean"]) / stats["f0_std"], 0.0).astype(np.float32)
        e_norm = ((e_gt - stats["energy_mean"]) / stats["energy_std"]).astype(np.float32)
        pitch_gt = torch.tensor(f0_norm, device=device).unsqueeze(0)
        energy_gt_t = torch.tensor(e_norm, device=device).unsqueeze(0)

        mel_model_norm = run_a0(model, phone_ids, dur_gt_t, pitch_gt, energy_gt_t)

        full_text = rec["text"]
        content = full_text.split("。", 1)[1] if "。" in full_text else full_text

        T = min(mel_model_norm.shape[0], mel_gt_norm.shape[0])

        # ── Per-bin L1 error ──
        per_bin_error = np.abs(mel_model_norm[:T] - mel_gt_norm[:T]).mean(axis=0)  # [80]

        # ── Per-frame L1 error ──
        per_frame_error = np.abs(mel_model_norm[:T] - mel_gt_norm[:T]).mean(axis=1)  # [T]

        # ── Error in raw mel space ──
        mel_model_raw = mel_model_norm * mel_std + mel_mean
        per_bin_error_raw = np.abs(mel_model_raw[:T] - mel_gt_raw[:T]).mean(axis=0)

        print(f"\n{'='*70}")
        print(f"{target_id}: {content[:40]}")
        print(f"{'='*70}")

        print(f"\n  Overall mel L1 (normalized): {np.abs(mel_model_norm[:T] - mel_gt_norm[:T]).mean():.4f}")
        print(f"  Overall mel L1 (raw):        {np.abs(mel_model_raw[:T] - mel_gt_raw[:T]).mean():.4f}")

        # Top-10 worst bins
        worst_bins = np.argsort(per_bin_error)[::-1][:10]
        print(f"\n  Top-10 worst mel bins (normalized L1):")
        for b in worst_bins:
            # Convert mel bin to approximate frequency
            # sr=24000, n_fft=2048, n_mels=80, fmin=80, fmax=7600
            # mel_scale: 80 bins from 80 to 7600 Hz
            freq = 80 + (7600 - 80) * b / 79  # rough linear approx
            print(f"    bin {b:3d} (~{freq:.0f}Hz): L1_norm={per_bin_error[b]:.4f}, L1_raw={per_bin_error_raw[b]:.4f}")

        # Frame-level error stats
        print(f"\n  Per-frame error: mean={per_frame_error.mean():.4f}, max={per_frame_error.max():.4f} (frame {per_frame_error.argmax()})")
        # Worst 10% frames
        n_worst = max(1, T // 10)
        worst_frames = np.argsort(per_frame_error)[::-1][:n_worst]
        print(f"  Worst {n_worst} frames (of {T}): mean L1 = {per_frame_error[worst_frames].mean():.4f}")

        # ── Vocoder comparison ──
        print(f"\n  Vocoder comparison (raw mel space):")

        # GT raw
        audio_gt = vocode(mel_gt_raw[:T], voc_sess)
        wav_gt = str(OUT / f"{target_id}_gt_raw.wav")
        sf.write(wav_gt, audio_gt, 24000)
        cer_gt = cer_eval(asr, wav_gt, content)

        # Model raw (denormalized)
        audio_model = vocode(mel_model_raw[:T], voc_sess)
        wav_model = str(OUT / f"{target_id}_model_raw.wav")
        sf.write(wav_model, audio_model, 24000)
        cer_model = cer_eval(asr, wav_model, content)

        # Model norm (current A0 path)
        audio_model_n = vocode(mel_model_norm[:T], voc_sess)
        wav_model_n = str(OUT / f"{target_id}_model_norm.wav")
        sf.write(wav_model_n, audio_model_n, 24000)
        cer_model_n = cer_eval(asr, wav_model_n, content)

        print(f"    GT raw → vocoder:       CER={cer_gt:.0%}")
        print(f"    Model raw → vocoder:    CER={cer_model:.0%}")
        print(f"    Model norm → vocoder:   CER={cer_model_n:.0%}")
        print(f"    Δ(model_raw, gt_raw) =  {cer_model - cer_gt:+.0%}  (model error contribution)")


if __name__ == "__main__":
    main()
