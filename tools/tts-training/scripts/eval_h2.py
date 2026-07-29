"""H2 evaluation: holdout curve (6k/12k/18k/24k) + train_171 + line-level A0.

For each holdout poem, evaluates:
  - Full-poem A0 (same as H1 eval)
  - Line-level A0: model generates each body line independently, then ASR

Line-level A0 isolates "can the model produce good local segments" from
"can it compose them into long sequences".

Comparison with H1 determines whether mixed-window training helps.
"""
import argparse, json, math, sys, time, numpy as np, torch, soundfile as sf, onnxruntime as ort
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
sys.path.insert(0, str(ROOT / "scripts"))
from nar_fastspeech2 import FastSpeech2
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail
from h2_linebound import compute_body_lines

HIFIGAN = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
STATS_PATH = str(ROOT / "data" / "train_171_norm_stats.json")
FEAT_DIR = str(ROOT / "data" / "paddle_distill_features")
TEACHER_DIR = str(ROOT / "data" / "paddle_mfa_corpus")
OUT = ROOT / "output" / "unified_eval"

with open(STATS_PATH) as f:
    stats = json.load(f)

voc_sess = ort.InferenceSession(
    str(HIFIGAN / "hifigan_csmsc.onnx"),
    providers=["CPUExecutionProvider"],
)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def run_a0(model, phone_ids_list, dur_list, f0_norm_list, e_norm_list):
    """Batch A0 inference. Returns list of normalized mel arrays."""
    results = []
    with torch.no_grad():
        for i in range(len(phone_ids_list)):
            pids = torch.tensor([phone_ids_list[i]], dtype=torch.long)
            dur = torch.tensor([dur_list[i]], dtype=torch.long)
            pitch = torch.tensor(f0_norm_list[i]).unsqueeze(0)
            energy = torch.tensor(e_norm_list[i]).unsqueeze(0)

            x = model.embedding(pids) * math.sqrt(model.d_model)
            x = model.pos_enc(x)
            for layer in model.encoder_layers:
                x = layer(x)
            mel_input = model.length_regulator(x, dur)
            T_out = mel_input.size(1)
            mel_input = mel_input + \
                model.pitch_embed(pitch[:, :T_out].unsqueeze(-1)) + \
                model.energy_embed(energy[:, :T_out].unsqueeze(-1))
            mel_input = model.pos_enc(mel_input)
            dec = mel_input
            for layer in model.decoder_layers:
                dec = layer(dec)
            mel = model.mel_linear(dec)[0].numpy()
            results.append(mel)
    return results


def eval_full_poem(model, rec, asr, out_dir, label_prefix=""):
    """Evaluate a full poem: P2 (reconstruction) + P3 (student A0)."""
    pid = rec["poem_id"]
    npz = np.load(str(Path(FEAT_DIR) / f"{pid}.npz"))
    mel_gt_raw = npz["mel"].astype(np.float32)
    f0_gt = npz["f0"].astype(np.float32)
    e_gt = npz["energy"].astype(np.float32)
    dur_gt = np.array(rec["durations"], dtype=np.int32)

    mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
    mel_std = np.array(stats["mel_std"], dtype=np.float32)

    f0_norm = np.where(
        f0_gt > 0,
        (np.log(np.maximum(f0_gt, 1)) - stats["f0_mean"]) / stats["f0_std"],
        0.0,
    ).astype(np.float32)
    e_norm = ((e_gt - stats["energy_mean"]) / stats["energy_std"]).astype(np.float32)

    mel_model_norm = run_a0(model, [rec["phoneme_ids"]], [dur_gt], [f0_norm], [e_norm])[0]
    mel_model_raw = mel_model_norm * mel_std + mel_mean
    T = min(mel_model_norm.shape[0], mel_gt_raw.shape[0])
    mel_gt_norm = (mel_gt_raw - mel_mean) / mel_std
    l1 = float(np.abs(mel_model_norm[:T] - mel_gt_norm[:T]).mean())

    full_text = rec["text"]

    # P2: GT mel → vocoder → ASR
    audio_gt = voc_sess.run(None, {"logmel": mel_gt_raw[:T].astype(np.float32)})[0].flatten()
    wav_gt = str(out_dir / f"{pid}_full_gt.wav")
    sf.write(wav_gt, audio_gt, 24000)
    asr2, _ = transcribe(asr, wav_gt)
    cer2, _, _, _, _ = cer_detail(full_text, asr2)

    # P3: model mel → vocoder → ASR
    audio_model = voc_sess.run(None, {"logmel": mel_model_raw[:T].astype(np.float32)})[0].flatten()
    wav_model = str(out_dir / f"{pid}_full_model.wav")
    sf.write(wav_model, audio_model, 24000)
    asr3, _ = transcribe(asr, wav_model)
    cer3, _, _, _, _ = cer_detail(full_text, asr3)

    delta = cer3 - cer2
    return {
        "poem_id": pid,
        "cer2": cer2, "cer3": cer3,
        "student_delta": delta,
        "mel_l1": l1,
        "mel_len": rec["mel_len"],
        "full_text": full_text,
    }


def eval_line_level(model, rec, asr, out_dir):
    """Evaluate each body line independently through A0 + vocoder + ASR.

    Returns per-line and aggregate results.
    """
    pid = rec["poem_id"]
    npz = np.load(str(Path(FEAT_DIR) / f"{pid}.npz"))
    mel_gt_raw = npz["mel"].astype(np.float32)
    f0_gt = npz["f0"].astype(np.float32)
    e_gt = npz["energy"].astype(np.float32)

    mel_mean = np.array(stats["mel_mean"], dtype=np.float32)
    mel_std = np.array(stats["mel_std"], dtype=np.float32)

    segs = compute_body_lines(rec["text"], rec["phoneme_ids"], rec["durations"])
    if not segs:
        return None

    line_results = []
    all_phone_ids = []
    all_durs = []
    all_f0 = []
    all_e = []

    for si, seg in enumerate(segs):
        ps, pe = seg["phone_start"], seg["phone_end"]
        fs, fe = seg["frame_start"], seg["frame_end"]
        seg_pids = rec["phoneme_ids"][ps:pe]
        seg_durs = rec["durations"][ps:pe]
        seg_f0 = f0_gt[fs:fe]
        seg_e = e_gt[fs:fe]
        seg_mel_gt = mel_gt_raw[fs:fe]

        seg_f0_norm = np.where(
            seg_f0 > 0,
            (np.log(np.maximum(seg_f0, 1)) - stats["f0_mean"]) / stats["f0_std"],
            0.0,
        ).astype(np.float32)
        seg_e_norm = ((seg_e - stats["energy_mean"]) / stats["energy_std"]).astype(np.float32)

        all_phone_ids.append(seg_pids)
        all_durs.append(seg_durs)
        all_f0.append(seg_f0_norm)
        all_e.append(seg_e_norm)

    mels_model = run_a0(model, all_phone_ids, all_durs, all_f0, all_e)

    for si, (seg, mel_model_norm) in enumerate(zip(segs, mels_model)):
        fs, fe = seg["frame_start"], seg["frame_end"]
        seg_mel_gt = mel_gt_raw[fs:fe]
        mel_model_raw = mel_model_norm * mel_std + mel_mean
        T = min(mel_model_norm.shape[0], seg_mel_gt.shape[0])
        seg_mel_gt_norm = (seg_mel_gt - mel_mean) / mel_std
        l1 = float(np.abs(mel_model_norm[:T] - seg_mel_gt_norm[:T]).mean())

        line_text = seg["text"].rstrip("，。？！；：")

        # P2: GT line mel → vocoder → ASR
        audio_gt = voc_sess.run(None, {"logmel": seg_mel_gt[:T].astype(np.float32)})[0].flatten()
        wav_gt = str(out_dir / f"{pid}_line{si}_gt.wav")
        sf.write(wav_gt, audio_gt, 24000)
        asr2, _ = transcribe(asr, wav_gt)
        cer2, _, _, _, _ = cer_detail(line_text, asr2)

        # P3: model line mel → vocoder → ASR
        audio_model = voc_sess.run(None, {"logmel": mel_model_raw[:T].astype(np.float32)})[0].flatten()
        wav_model = str(out_dir / f"{pid}_line{si}_model.wav")
        sf.write(wav_model, audio_model, 24000)
        asr3, _ = transcribe(asr, wav_model)
        cer3, _, _, _, _ = cer_detail(line_text, asr3)

        delta = cer3 - cer2
        line_results.append({
            "line_idx": si,
            "text": line_text,
            "cer2": cer2, "cer3": cer3,
            "student_delta": delta,
            "mel_l1": l1,
            "n_frames": seg["n_frames"],
        })

    return line_results


def summarize(results, label):
    if not results:
        return None
    deltas = [r["student_delta"] for r in results]
    abs_d = [abs(d) for d in deltas]
    l1s = [r["mel_l1"] for r in results]
    cers3 = [r["cer3"] for r in results]
    empty = sum(1 for r in results if r["cer3"] >= 0.99)

    s = {
        "n": len(results),
        "delta_mean": float(np.mean(deltas)),
        "delta_median": float(np.median(deltas)),
        "delta_p75": float(np.percentile(deltas, 75)),
        "delta_p90": float(np.percentile(deltas, 90)),
        "abs_delta_le5pp": sum(1 for d in abs_d if d <= 0.05),
        "abs_delta_le5pp_rate": sum(1 for d in abs_d if d <= 0.05) / len(results),
        "p3_100pct": empty,
        "p3_empty_rate": empty / len(results),
        "cer3_mean": float(np.mean(cers3)),
        "cer2_mean": float(np.mean([r["cer2"] for r in results])),
        "mel_l1_mean": float(np.mean(l1s)),
    }
    print(f"\n  [{label}] Summary ({s['n']} samples):")
    print(f"    delta: mean={s['delta_mean']:+.1%}  median={s['delta_median']:+.1%}  P75={s['delta_p75']:+.1%}")
    print(f"    |d|<=5pp: {s['abs_delta_le5pp']}/{s['n']}  P3=100%: {s['p3_100pct']}/{s['n']}")
    print(f"    mel L1: mean={s['mel_l1_mean']:.3f}")
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_labels", nargs="+", default=["6k", "12k", "18k", "24k"])
    parser.add_argument("--skip_line", action="store_true", help="Skip line-level eval")
    parser.add_argument("--skip_train", action="store_true", help="Skip train_171 eval")
    a = parser.parse_args()

    step_map = {
        "6k": "H2_step6000_slim.pt",
        "12k": "H2_step12000_slim.pt",
        "18k": "H2_step18000_slim.pt",
        "24k": "H2_step24000_slim.pt",
    }

    holdout = [json.loads(l) for l in open(ROOT / "data" / "holdout_20_manifest.jsonl", encoding="utf-8")]
    holdout.sort(key=lambda r: r["mel_len"])

    print("Loading ASR...")
    asr = load_asr_model()
    print("ASR loaded.\n")

    all_curves = {}

    for label in a.step_labels:
        ckpt_path = ROOT / "checkpoints" / step_map[label]
        if not ckpt_path.exists():
            print(f"SKIP {label}: {ckpt_path} not found")
            continue

        print(f"\n{'=' * 70}")
        print(f"H2 HOLDOUT_20 @ {label} (full poem)")
        print(f"{'=' * 70}")
        model, ckpt = load_model(str(ckpt_path))
        cum_frames = ckpt.get("cum_frames", "?")
        win_counts = ckpt.get("win_counts", "?")
        print(f"  cum_frames={cum_frames} win_counts={win_counts}")

        out_dir = OUT / f"H2_holdout20_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)

        full_results = []
        for idx, rec in enumerate(holdout):
            r = eval_full_poem(model, rec, asr, out_dir)
            full_results.append(r)
            if idx % 5 == 0 or idx < 3:
                print(f"  [{idx + 1}/{len(holdout)}] {r['poem_id']}: "
                      f"L1={r['mel_l1']:.3f} P2={r['cer2']:.0%} P3={r['cer3']:.0%} "
                      f"d={r['student_delta']:+.0%} | {r['full_text'][:25]}")

        s_full = summarize(full_results, f"holdout20_{label}_full")

        # Line-level evaluation
        line_summary = None
        if not a.skip_line:
            print(f"\n  --- Line-level A0 @ {label} ---")
            line_out = out_dir / "lines"
            line_out.mkdir(exist_ok=True)
            all_line_results = []
            for idx, rec in enumerate(holdout):
                lr = eval_line_level(model, rec, asr, line_out)
                if lr:
                    all_line_results.extend(lr)
                if idx % 5 == 0:
                    print(f"  [line eval {idx + 1}/{len(holdout)}]")

            line_summary = summarize(all_line_results, f"holdout20_{label}_line")

            # Key diagnostic
            if s_full and line_summary:
                print(f"\n  DIAGNOSTIC @ {label}:")
                print(f"    Full-poem delta:  {s_full['delta_mean']:+.1%}")
                print(f"    Line-level delta: {line_summary['delta_mean']:+.1%}")
                print(f"    Full P3=100%:     {s_full['p3_100pct']}/{s_full['n']}")
                print(f"    Line P3=100%:     {line_summary['p3_100pct']}/{line_summary['n']}")

        rpt = {
            "name": f"H2_holdout20_{label}",
            "step": label,
            "cum_frames": cum_frames,
            "win_counts": win_counts,
            "full_poem": {"summary": s_full, "samples": full_results},
            "line_level": {"summary": line_summary, "samples": None},
        }
        with open(out_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(rpt, f, ensure_ascii=False, indent=2)
        all_curves[label] = rpt

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"H2 HOLDOUT CURVE")
    print(f"{'=' * 70}")
    print(f"{'Step':>5} {'Full delta':>11} {'Line delta':>11} "
          f"{'Full P3=1':>9} {'Line P3=1':>9} {'L1 full':>7} {'L1 line':>7}")
    print("-" * 65)
    for label in a.step_labels:
        if label not in all_curves:
            continue
        r = all_curves[label]
        sf = r["full_poem"]["summary"]
        sl = r["line_level"]["summary"]
        fd = f"{sf['delta_mean']:+.1%}" if sf else "N/A"
        ld = f"{sl['delta_mean']:+.1%}" if sl else "N/A"
        fp = f"{sf['p3_100pct']}/{sf['n']}" if sf else "N/A"
        lp = f"{sl['p3_100pct']}/{sl['n']}" if sl else "N/A"
        fl = f"{sf['mel_l1_mean']:.3f}" if sf else "N/A"
        ll = f"{sl['mel_l1_mean']:.3f}" if sl else "N/A"
        print(f"{label:>5} {fd:>11} {ld:>11} {fp:>9} {lp:>9} {fl:>7} {ll:>7}")

    print("\nDONE")


if __name__ == "__main__":
    main()
