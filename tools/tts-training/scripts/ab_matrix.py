"""Complete A/B isolation matrix on Round 7 checkpoint.

Generates 8 audio variants (A0-A7) isolating each predictor's contribution.
Uses GT duration scaling to separate global length from local duration errors.
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
OUT = ROOT / "output" / "ab_matrix"
CKPT = ROOT / "checkpoints" / "fs2_final.pt"
POEM_ID = "poem_0285"
POEM_TEXT = "空山不见人，但闻人语响。返景入深林，复照青苔上。"

device = "cpu"


def load_model():
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stats = ckpt["stats"]
    return model, stats


def vocode(mel):
    sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))
    return sess.run(None, {"logmel": mel.astype(np.float32)})[0].flatten()


def scale_dur_to_gt(pred_dur, gt_dur):
    """Scale predicted durations to match GT total, distribute remainder."""
    scale = gt_dur.sum() / max(pred_dur.sum(), 1)
    scaled = np.round(pred_dur * scale).astype(int)
    diff = gt_dur.sum() - scaled.sum()
    if diff != 0:
        max_idx = np.argmax(scaled)
        scaled[max_idx] += diff
    return scaled


def run_forward(model, phone_ids, durations=None, pitches=None, energies=None):
    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)

        log_dur = model.duration_predictor(x)
        pitch_pred = model.pitch_predictor(x)
        energy_pred = model.energy_predictor(x)

        # Determine durations
        if durations is None:
            durations_t = log_dur.detach().exp().round().clamp(min=0).long()
        else:
            durations_t = durations

        mel_input = model.length_regulator(x, durations_t)
        T_out = mel_input.size(1)

        # Determine pitch
        if pitches is None:
            pitch_use = model.length_regulator(
                pitch_pred.unsqueeze(-1), durations_t
            ).squeeze(-1)
        else:
            pitch_use = pitches[:, :T_out] if pitches.size(1) >= T_out else \
                torch.cat([pitches, torch.zeros(1, T_out - pitches.size(1))], dim=1)

        # Determine energy
        if energies is None:
            energy_use = model.length_regulator(
                energy_pred.unsqueeze(-1), durations_t
            ).squeeze(-1)
        else:
            energy_use = energies[:, :T_out] if energies.size(1) >= T_out else \
                torch.cat([energies, torch.zeros(1, T_out - energies.size(1))], dim=1)

        pitch_embed = model.pitch_embed(pitch_use.unsqueeze(-1))
        energy_embed = model.energy_embed(energy_use.unsqueeze(-1))

        mel_input = mel_input[:, :T_out] + pitch_embed[:, :T_out] + energy_embed[:, :T_out]
        mel_input = model.pos_enc(mel_input)
        dec = mel_input
        for layer in model.decoder_layers:
            dec = layer(dec)
        mel = model.mel_linear(dec)
    return mel[0].numpy(), log_dur[0].exp().numpy(), pitch_pred[0].numpy(), energy_pred[0].numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    model, stats = load_model()
    mel_mean = np.array(stats["mel_mean"])
    mel_std = np.array(stats["mel_std"])
    f0_mean, f0_std = stats["f0_mean"], stats["f0_std"]
    e_mean, e_std = stats["energy_mean"], stats["energy_std"]

    # Load sample
    manifest = [json.loads(l) for l in open(ROOT / "data" / "train_manifest.jsonl", encoding="utf-8")]
    rec = next(r for r in manifest if r["poem_id"] == POEM_ID)
    npz = np.load(str(ROOT / rec["mel_path"]))
    mel_gt = npz["mel"]
    f0_gt = npz["f0"]
    energy_gt = npz["energy"]
    dur_gt = np.array(rec["durations"], dtype=np.int32)
    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long, device=device)

    T_gt = mel_gt.shape[0]
    dur_gt_t = torch.tensor([dur_gt], dtype=torch.long, device=device)

    # Prepare GT pitch/energy in normalized space (mel-length)
    f0_norm_gt = np.where(f0_gt > 0, (np.log(np.maximum(f0_gt, 1)) - f0_mean) / f0_std, 0.0).astype(np.float32)
    e_norm_gt = ((energy_gt - e_mean) / e_std).astype(np.float32)
    pitch_gt_t = torch.tensor(f0_norm_gt, device=device).unsqueeze(0)
    energy_gt_t = torch.tensor(e_norm_gt, device=device).unsqueeze(0)

    # Get raw predictions first
    _, pred_dur_raw, pred_pitch_raw, pred_energy_raw = run_forward(model, phone_ids)
    # pred_dur_raw is phoneme-level, need to expand to mel-level
    pred_dur_scaled = scale_dur_to_gt(pred_dur_raw, dur_gt)
    dur_pred_scaled_t = torch.tensor([pred_dur_scaled], dtype=torch.long, device=device)

    # Also get phoneme-level pred pitch/energy expanded via GT durations
    pred_pitch_phoneme = torch.tensor(pred_pitch_raw, device=device).unsqueeze(0)
    pred_energy_phoneme = torch.tensor(pred_energy_raw, device=device).unsqueeze(0)

    # Expand predictions to mel-length using GT durations
    pred_pitch_expanded = model.length_regulator(
        pred_pitch_phoneme.unsqueeze(-1), dur_gt_t
    ).squeeze(-1)
    pred_energy_expanded = model.length_regulator(
        pred_energy_phoneme.unsqueeze(-1), dur_gt_t
    ).squeeze(-1)

    # Global mean pitch/energy
    all_f0_voiced = f0_norm_gt[f0_gt > 0]
    global_mean_pitch = float(all_f0_voiced.mean()) if len(all_f0_voiced) > 0 else 0.0
    global_mean_energy = float(e_norm_gt.mean())
    pitch_flat = torch.full((1, T_gt), global_mean_pitch, device=device)
    energy_flat = torch.full((1, T_gt), global_mean_energy, device=device)

    # ── Define conditions ──
    conditions = [
        ("A0", dur_gt_t, pitch_gt_t, energy_gt_t, "GT all"),
        ("A1", dur_pred_scaled_t, pitch_gt_t, energy_gt_t, "scaled pred dur + GT pitch/energy"),
        ("A2", dur_gt_t, pred_pitch_expanded, energy_gt_t, "GT dur + pred pitch + GT energy"),
        ("A3", dur_gt_t, pitch_gt_t, pred_energy_expanded, "GT dur + GT pitch + pred energy"),
        ("A4", dur_gt_t, pred_pitch_expanded, pred_energy_expanded, "GT dur + pred pitch + pred energy"),
        ("A5", dur_pred_scaled_t, pred_pitch_expanded, pred_energy_expanded, "scaled pred dur + pred pitch/energy"),
        ("A6", dur_gt_t, pitch_flat, energy_flat, "GT dur + flat mean pitch/energy"),
        ("A7", None, None, None, "full prediction (no GT)"),
    ]

    print("Loading ASR model...")
    asr = load_asr_model()
    ref = normalize_text(POEM_TEXT)

    results = []
    for name, dur, pitch, energy, desc in conditions:
        print(f"\n{'='*60}")
        print(f"{name}: {desc}")

        mel, _, _, _ = run_forward(model, phone_ids, durations=dur, pitches=pitch, energies=energy)
        audio = vocode(mel)
        rms = np.sqrt(np.mean(audio**2))
        dur_sec = len(audio) / 24000

        wav_path = str(OUT / f"{name}.wav")
        sf.write(wav_path, audio, 24000)

        # ASR
        asr_text, _ = transcribe(asr, wav_path)
        hyp = normalize_text(asr_text)
        ops = align(ref, hyp)
        errs = sum(1 for r, h in ops if r != h)
        d = sum(1 for r, h in ops if r != '*' and h == '*')
        s = sum(1 for r, h in ops if r != '*' and h != '*' and r != h)
        i = sum(1 for r, h in ops if r == '*' and h != '*')
        cer = errs / max(len(ref), 1)

        status = "PASS" if cer < 0.3 else ("MARGINAL" if cer < 0.5 else "FAIL")

        print(f"  mel: {mel.shape}, range=[{mel.min():.2f}, {mel.max():.2f}]")
        print(f"  audio: {dur_sec:.1f}s, rms={rms:.4f}")
        print(f"  ASR: {asr_text}")
        print(f"  CER: {cer:.2%} D={d} S={s} I={i} → {status}")

        results.append({"name": name, "desc": desc, "cer": cer, "status": status, "asr": asr_text,
                        "mel_range": [float(mel.min()), float(mel.max())], "dur_sec": dur_sec, "rms": rms})

    # Summary
    print(f"\n{'='*60}")
    print(f"A/B Matrix Summary ({POEM_ID})")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['name']}: CER={r['cer']:.2%} {r['status']:8s} | {r['desc']}")
        print(f"        ASR: {r['asr']}")


if __name__ == "__main__":
    main()
