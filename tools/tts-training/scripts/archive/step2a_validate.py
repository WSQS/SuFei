"""Step 2a: Compare step12000 vs step14000 A0 on 12 validation samples.

For each sample: GT dur + GT pitch + GT energy → vocoder → ASR → CER.
Reports content CER, mel L1, audio duration for both checkpoints.
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
CKPT_DIR = ROOT / "checkpoints"
OUT = ROOT / "output" / "step2_validation"

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
    # Mel L1 (normalized space)
    mel_norm = (mel_target - mel_mean) / mel_std
    T_min = min(mel_np.shape[0], mel_norm.shape[0])
    mel_l1 = np.abs(mel_np[:T_min] - mel_norm[:T_min]).mean()
    return mel_np, mel_l1


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Select 12 diverse samples (short, medium, long, different forms)
    manifest = [json.loads(l) for l in open(ROOT / "data" / "train_manifest.jsonl", encoding="utf-8")]
    manifest.sort(key=lambda r: r["mel_len"])
    # Pick from across the length distribution
    indices = list(range(0, 24, 2))  # 0,2,4,...,22 → 12 samples
    samples = [manifest[i] for i in indices]

    # Load norm stats
    with open(ROOT / "data" / "norm_stats.json") as f:
        stats = json.load(f)
    mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
    mel_std = np.array(stats["mel_std"], dtype=np.float32)

    # Vocoder
    voc_sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))

    # ASR
    print("Loading ASR...")
    asr = load_asr_model()

    # Preload data for each sample
    data = []
    for rec in samples:
        npz = np.load(str(ROOT / rec["mel_path"]))
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

        # Reference text: strip title/author prefix
        full_text = rec["text"]
        content = full_text.split("。", 1)[1] if "。" in full_text else full_text
        ref = normalize_text(content)

        data.append({"poem_id": rec["poem_id"], "phone_ids": phone_ids, "dur_gt_t": dur_gt_t,
                     "pitch_gt": pitch_gt, "energy_gt_t": energy_gt_t, "mel_target": mel_gt,
                     "ref": ref, "mel_len": rec["mel_len"], "text": content[:30]})

    # Test both checkpoints
    ckpts = ["fs2_step12000", "fs2_step14000"]
    results = {ck: [] for ck in ckpts}

    for ckpt_name in ckpts:
        ckpt_path = CKPT_DIR / f"{ckpt_name}.pt"
        if not ckpt_path.exists():
            print(f"SKIP {ckpt_name}: not found")
            continue

        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = FastSpeech2(vocab_size=268, dropout=0.0)
        model.load_state_dict(ckpt["model"])
        model.eval()

        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_name}")

        for d in data:
            mel, mel_l1 = run_a0(model, d["phone_ids"], d["dur_gt_t"],
                                 d["pitch_gt"], d["energy_gt_t"], d["mel_target"], mel_mean, mel_std)
            audio = vocode(mel, voc_sess)
            wav_path = str(OUT / f"{ckpt_name}_{d['poem_id']}.wav")
            sf.write(wav_path, audio, 24000)

            asr_text, _ = transcribe(asr, wav_path)
            hyp = normalize_text(asr_text)
            ops = align(d["ref"], hyp)
            errs = sum(1 for r, h in ops if r != h)
            subs_d = sum(1 for r, h in ops if r != '*' and h == '*')
            subs_s = sum(1 for r, h in ops if r != '*' and h != '*' and r != h)
            subs_i = sum(1 for r, h in ops if r == '*' and h != '*')
            cer = (subs_d + subs_s) / max(len(d["ref"]), 1)

            dur_sec = len(audio) / 24000
            status = "OK" if cer < 0.15 else ("MARGINAL" if cer < 0.3 else "FAIL")

            print(f"  {d['poem_id']}: CER={cer:.0%} L1={mel_l1:.3f} dur={dur_sec:.1f}s {status} | {d['text']}")

            results[ckpt_name].append({"poem_id": d["poem_id"], "cer": cer,
                                       "mel_l1": mel_l1, "dur_sec": dur_sec, "status": status,
                                       "d": subs_d, "s": subs_s, "i": subs_i})

    # Summary
    print(f"\n{'='*60}")
    print(f"Validation Summary (12 samples)")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<16} {'Mean CER':>8} {'Median CER':>10} {'Pass':>6} {'Mean L1':>8}")
    print("-" * 50)
    for ck in ckpts:
        r = results[ck]
        if not r:
            continue
        cers = [x["cer"] for x in r]
        l1s = [x["mel_l1"] for x in r]
        n_pass = sum(1 for x in r if x["status"] == "OK")
        print(f"{ck:<16} {np.mean(cers):>8.1%} {np.median(cers):>10.1%} {n_pass:>3}/{len(r)}  {np.mean(l1s):>8.3f}")


if __name__ == "__main__":
    main()
