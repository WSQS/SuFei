"""Step 0: Cross-validate A/B matrix script.

Trains a single-sample overfit model (2000 steps, GT variance),
then runs A0 (GT all) through ab_matrix.py's code path.
Must reproduce CER=0% from the earlier manual test.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe, normalize_text, align
from nar_build_manifest import text_to_phonemes

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
OUT = ROOT / "output" / "ab_matrix"
POEM_ID = "poem_0285"
POEM_TEXT = "空山不见人，但闻人语响。返景入深林，复照青苔上。"

device = "cpu"


def vocode(mel):
    sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))
    return sess.run(None, {"logmel": mel.astype(np.float32)})[0].flatten()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Load sample
    manifest = [json.loads(l) for l in open(ROOT / "data" / "train_manifest.jsonl", encoding="utf-8")]
    rec = next(r for r in manifest if r["poem_id"] == POEM_ID)
    npz = np.load(str(ROOT / rec["mel_path"]))
    mel_gt = npz["mel"].astype(np.float32)
    f0_gt = npz["f0"].astype(np.float32)
    energy_gt = npz["energy"].astype(np.float32)
    dur_gt = np.array(rec["durations"], dtype=np.int32)
    phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long, device=device)
    dur_gt_t = torch.tensor(np.array([dur_gt]), dtype=torch.long, device=device)

    # Normalize
    with open(ROOT / "data" / "norm_stats.json") as f:
        stats = json.load(f)
    f0_mean, f0_std = stats["f0_mean"], stats["f0_std"]
    e_mean, e_std = stats["energy_mean"], stats["energy_std"]

    f0_norm = np.where(f0_gt > 0, (np.log(np.maximum(f0_gt, 1)) - f0_mean) / f0_std, 0.0).astype(np.float32)
    e_norm = ((energy_gt - e_mean) / e_std).astype(np.float32)
    pitch_gt = torch.tensor(f0_norm, device=device).unsqueeze(0)
    energy_gt_t = torch.tensor(e_norm, device=device).unsqueeze(0)
    mel_target = torch.tensor(mel_gt, device=device).unsqueeze(0)

    print(f"Sample: {POEM_ID}")
    print(f"  mel: {mel_gt.shape}, dur_sum: {dur_gt.sum()}, phones: {len(rec['phoneme_ids'])}")
    print(f"  f0_norm: mean={f0_norm.mean():.3f} std={f0_norm.std():.3f}")
    print(f"  e_norm: mean={e_norm.mean():.3f} std={e_norm.std():.3f}")

    # Train overfit (2000 steps, GT variance)
    print(f"\nTraining overfit model (2000 steps, GT dur+pitch+energy)...")
    model = FastSpeech2(vocab_size=268, dropout=0.0, predictor_dropout=0.0, mean_log_dur=2.7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1, 2001):
        model.train()
        optimizer.zero_grad()

        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)

        # Predictors (compute loss but DON'T inject into decoder)
        log_dur = model.duration_predictor(x)
        pitch_p = model.pitch_predictor(x)
        energy_p = model.energy_predictor(x)

        # Decoder uses GT variance
        mel_input = model.length_regulator(x, dur_gt_t)
        T_pred = mel_input.size(1)
        T_min = min(T_pred, mel_target.size(1))

        mel_input = mel_input + \
            model.pitch_embed(pitch_gt[:, :T_pred].unsqueeze(-1)) + \
            model.energy_embed(energy_gt_t[:, :T_pred].unsqueeze(-1))
        mel_input = model.pos_enc(mel_input)
        dec = mel_input
        for layer in model.decoder_layers:
            dec = layer(dec)
        mel_pred = model.mel_linear(dec)

        mel_loss = F.l1_loss(mel_pred[:, :T_min], mel_target[:, :T_min])
        mel_loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == 1:
            print(f"  step {step}: mel={mel_loss.item():.4f}")

    model.eval()

    # ── A0: GT all ──
    print(f"\n{'='*60}")
    print(f"A0 (overfit checkpoint, GT all):")

    with torch.no_grad():
        x = model.embedding(phone_ids) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)

        mel_input = model.length_regulator(x, dur_gt_t)
        T_out = mel_input.size(1)

        pitch_use = pitch_gt[:, :T_out]
        energy_use = energy_gt_t[:, :T_out]

        mel_input = mel_input + \
            model.pitch_embed(pitch_use.unsqueeze(-1)) + \
            model.energy_embed(energy_use.unsqueeze(-1))
        mel_input = model.pos_enc(mel_input)
        dec = mel_input
        for layer in model.decoder_layers:
            dec = layer(dec)
        mel = model.mel_linear(dec)

    mel_np = mel[0].numpy()
    audio = vocode(mel_np)
    rms = np.sqrt(np.mean(audio**2))
    dur_sec = len(audio) / 24000

    print(f"  mel: {mel_np.shape}, range=[{mel_np.min():.2f}, {mel_np.max():.2f}]")
    print(f"  audio: {dur_sec:.1f}s, rms={rms:.4f}")

    wav_path = str(OUT / "overfit_A0.wav")
    sf.write(wav_path, audio, 24000)

    # ASR
    print(f"\nLoading ASR...")
    asr = load_asr_model()
    ref = normalize_text(POEM_TEXT)
    asr_text, _ = transcribe(asr, wav_path)
    hyp = normalize_text(asr_text)
    ops = align(ref, hyp)
    errs = sum(1 for r, h in ops if r != h)
    d = sum(1 for r, h in ops if r != '*' and h == '*')
    s = sum(1 for r, h in ops if r != '*' and h != '*' and r != h)
    i = sum(1 for r, h in ops if r == '*' and h != '*')
    cer = errs / max(len(ref), 1)

    print(f"  ASR: {asr_text}")
    print(f"  CER: {cer:.2%} D={d} S={s} I={i}")
    print(f"  Expected: CER=30% (6 insertions from title prefix)")
    print(f"  Content CER (excluding prefix): {'0%' if d == 0 and s == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
