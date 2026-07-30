"""A/B test: GT duration vs predicted duration on overfit model.

Key question: with GT durations, does the acoustic decoder produce
intelligible mel? This isolates duration predictor from decoder.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nar_fastspeech2 import FastSpeech2

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
OUT = ROOT / "output" / "ab_test"
POEM_TEXT = "鹿柴，唐代·王维。空山不见人，但闻人语响。返景入深林，复照青苔上。"


def vocode(mel):
    sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))
    return sess.run(None, {"logmel": mel.astype(np.float32)})[0].flatten()


def main():
    device = "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    # Load sample
    manifest = [json.loads(l) for l in open(ROOT / "data" / "train_manifest.jsonl", encoding="utf-8")]
    rec = next(r for r in manifest if r["poem_id"] == "poem_0285")
    npz = np.load(str(ROOT / rec["mel_path"]))

    mel_gt = npz["mel"]
    f0_gt = npz["f0"]
    energy_gt = npz["energy"]
    durations_gt = np.array(rec["durations"], dtype=np.int32)
    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long, device=device)
    durations_t = torch.tensor([durations_gt], dtype=torch.long, device=device)

    print(f"Sample: {rec['poem_id']}")
    print(f"  mel: {mel_gt.shape}, dur sum: {durations_gt.sum()}, phones: {len(rec['phoneme_ids'])}")
    print(f"  text: {POEM_TEXT[:50]}")

    # ─── Train overfit model ───
    print(f"\nTraining overfit model (3000 steps)...")
    model = FastSpeech2(vocab_size=268, dropout=0.0, mean_log_dur=2.7).to(device)

    mel_target = torch.tensor(mel_gt, dtype=torch.float32, device=device).unsqueeze(0)
    f0_norm = torch.where(
        torch.tensor(f0_gt) > 0,
        (torch.log(torch.tensor(f0_gt).clamp(min=1)) - 5.34) / 0.44,
        torch.zeros_like(torch.tensor(f0_gt)),
    ).float().to(device).unsqueeze(0)
    energy_norm = ((torch.tensor(energy_gt) - (-633.79)) / 216.68).float().to(device).unsqueeze(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, 2001):
        model.train()
        optimizer.zero_grad()

        x = model.embedding(phone_ids) * np.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)

        log_dur_pred = model.duration_predictor(x)
        pitch_pred = model.pitch_predictor(x)
        energy_pred = model.energy_predictor(x)

        mel_input = model.length_regulator(x, durations_t)
        T_pred = mel_input.size(1)
        T_gt = mel_target.size(1)
        T_min = min(T_pred, T_gt)

        pitch_exp = model.length_regulator(pitch_pred.unsqueeze(-1), durations_t).squeeze(-1)
        energy_exp = model.length_regulator(energy_pred.unsqueeze(-1), durations_t).squeeze(-1)

        mel_input = mel_input + model.pitch_embed(f0_norm[:, :T_pred].unsqueeze(-1)) + model.energy_embed(energy_norm[:, :T_pred].unsqueeze(-1))
        mel_input = model.pos_enc(mel_input)
        dec = mel_input
        for layer in model.decoder_layers:
            dec = layer(dec)
        mel_pred = model.mel_linear(dec)

        mel_loss = F.l1_loss(mel_pred[:, :T_min], mel_target[:, :T_min])
        log_dur_gt = torch.log(durations_t.float().clamp(min=1, max=100))
        dur_loss = F.l1_loss(log_dur_pred, log_dur_gt)
        total = mel_loss + dur_loss
        total.backward()
        optimizer.step()

        if step % 500 == 0 or step == 1:
            pred_dur_sum = log_dur_pred.exp().sum().item()
            print(f"  step {step}: mel={mel_loss:.4f} dur={dur_loss:.4f} pred_dur_sum={pred_dur_sum:.0f}")

    model.eval()

    # ─── A: GT duration inference ───
    print(f"\n=== A: GT duration ===")
    with torch.no_grad():
        mel_a, _, _, _ = model(phone_ids, durations=durations_t)
    mel_a_np = mel_a[0].cpu().numpy()
    audio_a = vocode(mel_a_np)
    dur_a = len(audio_a) / 24000
    print(f"  mel: {mel_a_np.shape}, audio: {dur_a:.1f}s, rms={np.sqrt(np.mean(audio_a**2)):.4f}")
    sf.write(str(OUT / "A_gt_dur.wav"), audio_a, 24000)

    # ─── B: Predicted duration ───
    print(f"\n=== B: Predicted duration ===")
    with torch.no_grad():
        mel_b, log_dur_b, _, _ = model(phone_ids)
    pred_dur_b = log_dur_b[0].exp()
    pred_sum_b = pred_dur_b.sum().item()
    mel_b_np = mel_b[0].cpu().numpy()
    audio_b = vocode(mel_b_np)
    dur_b = len(audio_b) / 24000
    scale_needed = durations_gt.sum() / pred_sum_b if pred_sum_b > 0 else 0
    print(f"  pred dur sum: {pred_sum_b:.0f} (GT: {durations_gt.sum()})")
    print(f"  ratio: {pred_sum_b/durations_gt.sum():.2%}, scale needed: {scale_needed:.2f}")
    print(f"  mel: {mel_b_np.shape}, audio: {dur_b:.1f}s, rms={np.sqrt(np.mean(audio_b**2)):.4f}")
    sf.write(str(OUT / "B_pred_dur.wav"), audio_b, 24000)

    # ─── C: Predicted duration × scale ───
    print(f"\n=== C: Predicted duration × {scale_needed:.2f} ===")
    scaled_dur = (pred_dur_b * scale_needed).round().clamp(min=1).long().unsqueeze(0)
    with torch.no_grad():
        mel_c, _, _, _ = model(phone_ids, durations=scaled_dur.to(device))
    mel_c_np = mel_c[0].cpu().numpy()
    audio_c = vocode(mel_c_np)
    dur_c = len(audio_c) / 24000
    print(f"  scaled dur sum: {scaled_dur.sum().item()}")
    print(f"  mel: {mel_c_np.shape}, audio: {dur_c:.1f}s, rms={np.sqrt(np.mean(audio_c**2)):.4f}")
    sf.write(str(OUT / "C_scaled_dur.wav"), audio_c, 24000)

    print(f"\nDone. Audio saved to {OUT}")
    print(f"  A_gt_dur.wav — GT duration ({dur_a:.1f}s)")
    print(f"  B_pred_dur.wav — predicted duration ({dur_b:.1f}s)")
    print(f"  C_scaled_dur.wav — scaled ×{scale_needed:.2f} ({dur_c:.1f}s)")


if __name__ == "__main__":
    main()