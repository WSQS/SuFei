"""B' A0 Validation: GT dur + GT pitch + GT energy → vocoder → ASR → CER.

Compares B' (PaddleSpeech distillation) against A (CosyVoice) baseline.
12 validation samples from paddle_distill_manifest, sorted by mel_len.
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
OUT = ROOT / "output" / "bprime_validation"

device = "cpu"


def vocode(mel, sess):
    return sess.run(None, {"logmel": mel.astype(np.float32)})[0].flatten()


def run_a0(model, phone_ids, dur_gt_t, pitch_gt, energy_gt_t, mel_target, mel_mean, mel_std):
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

    mel_np = mel[0].numpy()
    mel_norm = (mel_target - mel_mean) / mel_std
    T_min = min(mel_np.shape[0], mel_norm.shape[0])
    mel_l1 = np.abs(mel_np[:T_min] - mel_norm[:T_min]).mean()
    return mel_np, mel_l1


def evaluate_checkpoint(ckpt_path, ckpt_name, data, stats, voc_sess, asr, mel_mean, mel_std):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_name}")
    print(f"{'='*60}")

    results = []
    for d in data:
        mel, mel_l1 = run_a0(model, d["phone_ids"], d["dur_gt_t"],
                             d["pitch_gt"], d["energy_gt_t"], d["mel_target"], mel_mean, mel_std)
        audio = vocode(mel, voc_sess)
        wav_path = str(OUT / f"{ckpt_name}_{d['poem_id']}.wav")
        sf.write(wav_path, audio, 24000)

        asr_text, _ = transcribe(asr, wav_path)
        hyp = normalize_text(asr_text)
        ops = align(d["ref"], hyp)
        subs_d = sum(1 for r, h in ops if r != '*' and h == '*')
        subs_s = sum(1 for r, h in ops if r != '*' and h != '*' and r != h)
        subs_i = sum(1 for r, h in ops if r == '*' and h != '*')
        cer = (subs_d + subs_s) / max(len(d["ref"]), 1)

        dur_sec = len(audio) / 24000
        status = "OK" if cer < 0.15 else ("MARGINAL" if cer < 0.3 else "FAIL")

        print(f"  {d['poem_id']}: CER={cer:.0%} L1={mel_l1:.3f} dur={dur_sec:.1f}s {status} | {d['text']}")

        results.append({"poem_id": d["poem_id"], "cer": cer,
                        "mel_l1": float(mel_l1), "dur_sec": dur_sec, "status": status})

    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # B' manifest and stats
    manifest = [json.loads(l) for l in open(ROOT / "data" / "paddle_distill_manifest.jsonl", encoding="utf-8")]
    manifest.sort(key=lambda r: r["mel_len"])
    indices = list(range(0, 24, 2))  # 12 samples from length distribution
    samples = [manifest[i] for i in indices if i < len(manifest)]

    with open(ROOT / "data" / "paddle_distill_norm_stats.json") as f:
        stats = json.load(f)
    mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
    mel_std = np.array(stats["mel_std"], dtype=np.float32)

    voc_sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))

    print("Loading ASR...")
    asr = load_asr_model()

    # Preload data
    data = []
    for rec in samples:
        npz_path = ROOT / "data" / "paddle_distill_features" / f"{rec['poem_id']}.npz"
        if not npz_path.exists():
            print(f"  SKIP {rec['poem_id']}: no features")
            continue
        npz = np.load(str(npz_path))
        mel_gt = npz["mel"].astype(np.float32)
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
        ref = normalize_text(content)

        data.append({"poem_id": rec["poem_id"], "phone_ids": phone_ids, "dur_gt_t": dur_gt_t,
                     "pitch_gt": pitch_gt, "energy_gt_t": energy_gt_t, "mel_target": mel_gt,
                     "ref": ref, "text": content[:30]})

    print(f"\nValidation samples: {len(data)}")

    # Evaluate B' checkpoint
    bprime_results = evaluate_checkpoint(
        ROOT / "checkpoints" / "fs2_bprime_final.pt",
        "B_prime_step12000",
        data, stats, voc_sess, asr, mel_mean, mel_std
    )

    # Also evaluate A baseline if available
    a_ckpts = ["acoustic_baseline_step12000", "fs2_step12000"]
    a_results = None
    for a_name in a_ckpts:
        a_path = ROOT / "checkpoints" / f"{a_name}.pt"
        if a_path.exists():
            # A uses different norm stats
            a_stats_path = ROOT / "data" / "norm_stats.json"
            if a_stats_path.exists():
                with open(a_stats_path) as f:
                    a_stats = json.load(f)
                a_mel_mean = np.array(a_stats["mel_mean"], dtype=np.float32)
                a_mel_std = np.array(a_stats["mel_std"], dtype=np.float32)

                # A uses different manifest and features
                a_manifest = [json.loads(l) for l in open(ROOT / "data" / "train_manifest.jsonl", encoding="utf-8")]
                a_manifest.sort(key=lambda r: r["mel_len"])
                a_samples = [a_manifest[i] for i in indices if i < len(a_manifest)]

                a_data = []
                for rec in a_samples:
                    npz_path = ROOT / "data" / "nar_features" / f"{rec['poem_id']}.npz"
                    if not npz_path.exists():
                        continue
                    npz = np.load(str(npz_path))
                    mel_gt = npz["mel"].astype(np.float32)
                    f0_gt = npz["f0"].astype(np.float32)
                    e_gt = npz["energy"].astype(np.float32)
                    dur_gt = np.array(rec["durations"], dtype=np.int32)
                    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long, device=device)
                    dur_gt_t = torch.tensor(np.array([dur_gt]), dtype=torch.long, device=device)
                    f0_norm = np.where(f0_gt > 0, (np.log(np.maximum(f0_gt, 1)) - a_stats["f0_mean"]) / a_stats["f0_std"], 0.0).astype(np.float32)
                    e_norm = ((e_gt - a_stats["energy_mean"]) / a_stats["energy_std"]).astype(np.float32)
                    pitch_gt = torch.tensor(f0_norm, device=device).unsqueeze(0)
                    energy_gt_t = torch.tensor(e_norm, device=device).unsqueeze(0)
                    full_text = rec["text"]
                    content = full_text.split("。", 1)[1] if "。" in full_text else full_text
                    ref = normalize_text(content)
                    a_data.append({"poem_id": rec["poem_id"], "phone_ids": phone_ids, "dur_gt_t": dur_gt_t,
                                  "pitch_gt": pitch_gt, "energy_gt_t": energy_gt_t, "mel_target": mel_gt,
                                  "ref": ref, "text": content[:30]})

                if a_data:
                    print(f"\nLoading A baseline ({a_name})...")
                    a_results = evaluate_checkpoint(a_path, f"A_{a_name}", a_data, a_stats, voc_sess, asr, a_mel_mean, a_mel_std)
                    break

    # Summary
    print(f"\n{'='*60}")
    print(f"A0 Validation Summary")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<24} {'Mean CER':>8} {'Median CER':>10} {'Pass <15%':>9} {'Mean L1':>8}")
    print("-" * 60)

    for name, results in [("B' (PaddleSpeech)", bprime_results), ("A (CosyVoice)", a_results)]:
        if not results:
            continue
        cers = [x["cer"] for x in results]
        l1s = [x["mel_l1"] for x in results]
        n_pass = sum(1 for x in results if x["status"] == "OK")
        print(f"{name:<24} {np.mean(cers):>8.1%} {np.median(cers):>10.1%} {n_pass:>3}/{len(results)}    {np.mean(l1s):>8.3f}")


if __name__ == "__main__":
    main()
