"""Complete 4-path evaluation matrix with ASR transcripts.

For each poem, records:
  Path 1: 原始教师 wav → ASR           (baseline upper bound)
  Path 2: GT raw mel → HiFiGAN → ASR   (mel-vocoder reconstruction)
  Path 3: Model raw mel → HiFiGAN → ASR (student contribution)
  Path 4: Model norm mel → HiFiGAN → ASR (denorm bug demonstration)

Reports:
  - Full ASR transcript for each path
  - CER vs reference text
  - Reconstruction loss = Path2 - Path1
  - Student loss = Path3 - Path2
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
TEACHER_WAV = ROOT / "data" / "paddle_mfa_corpus"
OUT = ROOT / "output" / "matrix_eval"

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


def cer_detail(ref_text, hyp_text):
    ref = normalize_text(ref_text)
    hyp = normalize_text(hyp_text)
    ops = align(ref, hyp)
    subs_d = sum(1 for r, h in ops if r != '*' and h == '*')
    subs_s = sum(1 for r, h in ops if r != '*' and h != '*' and r != h)
    subs_i = sum(1 for r, h in ops if r == '*' and h != '*')
    cer = (subs_d + subs_s) / max(len(ref), 1)
    return cer, subs_d, subs_s, subs_i, len(ref)


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

    targets = ["poem_0244", "poem_0097", "poem_0298", "poem_0285"]

    results = []

    for target_id in targets:
        rec = next((r for r in manifest if r["poem_id"] == target_id), None)
        if rec is None:
            continue

        npz = np.load(str(ROOT / "data" / "paddle_distill_features" / f"{target_id}.npz"))
        mel_gt_raw = npz["mel"].astype(np.float32)
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
        mel_model_raw = mel_model_norm * mel_std + mel_mean

        full_text = rec["text"]
        content = full_text.split("。", 1)[1] if "。" in full_text else full_text
        T = min(mel_model_norm.shape[0], mel_gt_raw.shape[0])

        # ── Path 1: 原始教师 wav → ASR ──
        teacher_wav = TEACHER_WAV / f"{target_id}.wav"
        if teacher_wav.exists():
            asr_text_1, dur_1 = transcribe(asr, str(teacher_wav))
        else:
            asr_text_1, dur_1 = "[NOT FOUND]", 0

        # ── Path 2: GT raw mel → HiFiGAN → ASR ──
        audio_gt = vocode(mel_gt_raw[:T], voc_sess)
        wav_gt = str(OUT / f"{target_id}_gt_raw.wav")
        sf.write(wav_gt, audio_gt, 24000)
        asr_text_2, dur_2 = transcribe(asr, wav_gt)

        # ── Path 3: Model raw mel → HiFiGAN → ASR ──
        audio_model = vocode(mel_model_raw[:T], voc_sess)
        wav_model = str(OUT / f"{target_id}_model_raw.wav")
        sf.write(wav_model, audio_model, 24000)
        asr_text_3, dur_3 = transcribe(asr, wav_model)

        # ── Path 4: Model norm mel → HiFiGAN → ASR ──
        audio_model_n = vocode(mel_model_norm[:T], voc_sess)
        wav_model_n = str(OUT / f"{target_id}_model_norm.wav")
        sf.write(wav_model_n, audio_model_n, 24000)
        asr_text_4, dur_4 = transcribe(asr, wav_model_n)

        # ── CER ──
        cer1, d1, s1, i1, n1 = cer_detail(content, asr_text_1)
        cer2, d2, s2, i2, n2 = cer_detail(content, asr_text_2)
        cer3, d3, s3, i3, n3 = cer_detail(content, asr_text_3)
        cer4, d4, s4, i4, n4 = cer_detail(content, asr_text_4)

        recon_loss = cer2 - cer1
        student_loss = cer3 - cer2
        denorm_bug = cer4 - cer3

        print(f"\n{'='*70}")
        print(f"{target_id}: {content}")
        print(f"{'='*70}")
        print(f"\n  参考:           {content}")
        print(f"\n  Path 1 教师wav:  CER={cer1:.0%}  \"{asr_text_1}\"")
        print(f"  Path 2 GT重建:   CER={cer2:.0%}  \"{asr_text_2}\"")
        print(f"  Path 3 模型重建: CER={cer3:.0%}  \"{asr_text_3}\"")
        print(f"  Path 4 norm错误: CER={cer4:.0%}  \"{asr_text_4}\"")
        print(f"\n  重建损失 (P2-P1):  {recon_loss:+.0%}")
        print(f"  学生损失 (P3-P2):  {student_loss:+.0%}")
        print(f"  norm bug (P4-P3):  {denorm_bug:+.0%}")

        results.append({
            "poem_id": target_id, "content": content,
            "cer1": cer1, "cer2": cer2, "cer3": cer3, "cer4": cer4,
            "recon_loss": recon_loss, "student_loss": student_loss,
            "asr1": asr_text_1, "asr2": asr_text_2, "asr3": asr_text_3, "asr4": asr_text_4,
        })

    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"{'Poem':<14} {'教师wav':>8} {'GT重建':>8} {'模型重建':>8} {'norm错误':>8} {'重建损失':>8} {'学生损失':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['poem_id']:<14} {r['cer1']:>7.0%} {r['cer2']:>8.0%} {r['cer3']:>8.0%} {r['cer4']:>8.0%} "
              f"{r['recon_loss']:>+8.0%} {r['student_loss']:>+8.0%}")
    print(f"\n{'Mean':<14} {np.mean([r['cer1'] for r in results]):>7.0%} "
          f"{np.mean([r['cer2'] for r in results]):>8.0%} "
          f"{np.mean([r['cer3'] for r in results]):>8.0%} "
          f"{np.mean([r['cer4'] for r in results]):>8.0%} "
          f"{np.mean([r['recon_loss'] for r in results]):>+8.0%} "
          f"{np.mean([r['student_loss'] for r in results]):>+8.0%}")


if __name__ == "__main__":
    main()
