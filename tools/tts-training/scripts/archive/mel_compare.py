"""Compare GT mel vocoder output vs model mel vocoder output."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"


def vocode(mel):
    sess = ort.InferenceSession(str(HIFIGAN / "hifigan_csmsc.onnx"))
    return sess.run(None, {"logmel": mel.astype(np.float32)})[0].flatten()


def main():
    # Load GT mel for poem_0285 (鹿柴)
    gt_mel = np.load(str(ROOT / "output" / "overfit" / "poem_0285_mel_gt.npy"))
    pred_mel = np.load(str(ROOT / "output" / "overfit" / "poem_0285_mel_pred.npy"))

    print(f"GT mel: {gt_mel.shape}, range=[{gt_mel.min():.2f}, {gt_mel.max():.2f}], mean={gt_mel.mean():.2f}")
    print(f"Pred mel (overfit): {pred_mel.shape}, range=[{pred_mel.min():.2f}, {pred_mel.max():.2f}], mean={pred_mel.mean():.2f}")

    # Vocode GT
    gt_audio = vocode(gt_mel)
    print(f"\nGT audio: {len(gt_audio)} samples, {len(gt_audio)/24000:.1f}s, rms={np.sqrt(np.mean(gt_audio**2)):.4f}")

    # Vocode overfit pred
    pred_audio = vocode(pred_mel)
    print(f"Pred audio: {len(pred_audio)} samples, {len(pred_audio)/24000:.1f}s, rms={np.sqrt(np.mean(pred_audio**2)):.4f}")

    # Save
    out = ROOT / "output" / "mel_compare"
    out.mkdir(parents=True, exist_ok=True)
    sf.write(str(out / "gt_mel_vocoded.wav"), gt_audio, 24000)
    sf.write(str(out / "pred_mel_vocoded.wav"), pred_audio, 24000)

    # Now check step4000 model output
    # We need to load the mel from inference output
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    from nar_fastspeech2 import FastSpeech2
    from nar_build_manifest import text_to_phonemes

    phone_map = {}
    with open(ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0" / "phone_id_map.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                phone_map[parts[0]] = int(parts[1])

    ckpt = torch.load(str(ROOT / "checkpoints" / "fs2_step4000.pt"), map_location="cuda", weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0).to("cuda")
    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = ckpt["stats"]
    mel_mean = torch.tensor(stats["mel_mean"], device="cuda")
    mel_std = torch.tensor(stats["mel_std"], device="cuda")

    text = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
    phonemes = text_to_phonemes(text)
    phone_ids = [phone_map[p] for p, _ in phonemes if p in phone_map]
    phone_ids = torch.tensor([phone_ids], device="cuda")

    with torch.no_grad():
        mel_out, log_dur, _, _ = model(phone_ids)

    mel_norm = mel_out[0].cpu().numpy()
    mel_denorm = mel_norm * np.array(stats["mel_std"]) + np.array(stats["mel_mean"])

    print(f"\nStep4000 mel (norm): range=[{mel_norm.min():.2f}, {mel_norm.max():.2f}], mean={mel_norm.mean():.2f}")
    print(f"Step4000 mel (denorm): range=[{mel_denorm.min():.2f}, {mel_denorm.max():.2f}], mean={mel_denorm.mean():.2f}")

    audio = vocode(mel_denorm)
    print(f"Step4000 audio: {len(audio)} samples, rms={np.sqrt(np.mean(audio**2)):.4f}")
    sf.write(str(out / "step4000_mel_vocoded.wav"), audio, 24000)

    # Also save mel as npy for inspection
    np.save(str(out / "step4000_mel_denorm.npy"), mel_denorm)

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
