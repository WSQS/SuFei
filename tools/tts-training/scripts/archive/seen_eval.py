"""Seen-poem A0 evaluation: GT dur + GT pitch + GT energy -> vocoder -> ASR -> CER.

Evaluates how well each checkpoint performs on poems SEEN during training,
to distinguish memorization vs generalization.

B' uses paddle_distill_manifest + paddle_distill_features + paddle_distill_norm_stats.
A uses train_manifest + nar_features + norm_stats.
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
OUT = ROOT / "output" / "seen_validation"

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


def load_dataset(manifest_path, features_dir, stats_path, n_samples=12, offset=0):
    """Load n_samples from manifest starting at offset, sorted by mel_len."""
    manifest = [json.loads(l) for l in open(ROOT / "data" / manifest_path, encoding="utf-8")]
    manifest.sort(key=lambda r: r["mel_len"])
    samples = manifest[offset:offset + n_samples]

    with open(ROOT / "data" / stats_path) as f:
        stats = json.load(f)

    data = []
    for rec in samples:
        npz_path = ROOT / "data" / features_dir / f"{rec['poem_id']}.npz"
        if not npz_path.exists():
            print(f"  SKIP {rec['poem_id']}: no features at {npz_path}")
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

        data.append({
            "poem_id": rec["poem_id"], "phone_ids": phone_ids, "dur_gt_t": dur_gt_t,
            "pitch_gt": pitch_gt, "energy_gt_t": energy_gt_t,
            "ref": ref, "text": content[:40],
        })

    return data, stats


def evaluate_checkpoint(ckpt_path, ckpt_name, data, voc_sess, asr):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_name}")
    print(f"{'='*60}")

    results = []
    for d in data:
        mel_np = run_a0(model, d["phone_ids"], d["dur_gt_t"], d["pitch_gt"], d["energy_gt_t"])
        audio = vocode(mel_np, voc_sess)
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

        print(f"  {d['poem_id']}: CER={cer:.0%} dur={dur_sec:.1f}s {status} | {d['text']}")

        results.append({"poem_id": d["poem_id"], "cer": cer,
                        "dur_sec": dur_sec, "status": status})

    return results


def print_summary(name, results):
    cers = [x["cer"] for x in results]
    n_pass = sum(1 for x in results if x["status"] == "OK")
    n_marginal = sum(1 for x in results if x["status"] == "MARGINAL")
    print(f"{name:<24} {np.mean(cers):>8.1%} {np.median(cers):>10.1%} "
          f"{n_pass:>3}/{len(results)}    {n_marginal:>3}/{len(results)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    voc_sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))

    print("Loading ASR...")
    asr = load_asr_model()

    N = 12

    # === B' seen data (from training set) ===
    print("\n--- Loading B' seen samples (paddle_distill) ---")
    bp_data, bp_stats = load_dataset(
        "paddle_distill_manifest.jsonl", "paddle_distill_features",
        "paddle_distill_norm_stats.json", n_samples=N
    )
    print(f"  Loaded {len(bp_data)} samples")

    # === A seen data (from training set) ===
    print("\n--- Loading A seen samples (train_manifest) ---")
    a_data, a_stats = load_dataset(
        "train_manifest.jsonl", "nar_features",
        "norm_stats.json", n_samples=N
    )
    print(f"  Loaded {len(a_data)} samples")

    # Evaluate B'
    bp_ckpt = ROOT / "checkpoints" / "fs2_bprime_final.pt"
    bp_results = None
    if bp_ckpt.exists():
        bp_results = evaluate_checkpoint(bp_ckpt, "B_prime_seen", bp_data, voc_sess, asr)
    else:
        print(f"\nB' checkpoint not found: {bp_ckpt}")

    # Evaluate A
    a_ckpt = None
    for name in ["acoustic_baseline_step12000", "fs2_step12000"]:
        p = ROOT / "checkpoints" / f"{name}.pt"
        if p.exists():
            a_ckpt = p
            break
    a_results = None
    if a_ckpt:
        a_results = evaluate_checkpoint(a_ckpt, "A_seen", a_data, voc_sess, asr)
    else:
        print(f"\nA checkpoint not found")

    # Summary
    print(f"\n{'='*60}")
    print(f"Seen-Poem A0 Validation Summary")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<24} {'Mean CER':>8} {'Median CER':>10} {'Pass <15%':>9} {'Marginal':>9}")
    print("-" * 64)
    if bp_results:
        print_summary("B' (PaddleSpeech)", bp_results)
    if a_results:
        print_summary("A (from scratch)", a_results)


if __name__ == "__main__":
    main()
