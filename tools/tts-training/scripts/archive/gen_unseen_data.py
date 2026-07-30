"""Generate distillation features for unseen test poems.

For each unseen poem:
  1. Format text as "标题，朝代·作者。正文"
  2. G2P → phoneme IDs (same as training data)
  3. PaddleSpeech FS2 → teacher mel
  4. Teacher mel → HiFiGAN → teacher audio
  5. Teacher audio → mel/f0/energy extraction
  6. Save .npz features
  7. MFA align → durations
  8. Write unseen manifest

Output:
  data/unseen_features/{poem_id}.npz
  data/unseen_manifest.jsonl
  data/unseen_mfa_corpus/
  data/unseen_mfa_aligned/
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
sys.path.insert(0, str(ROOT / "scripts"))

from generate_paddlespeech_distillation_data import (
    text_to_phonemes, phones_to_ids, load_phone_id_map,
    run_fs2, run_hifigan, extract_mel, extract_f0, extract_energy,
    run_mfa_align, parse_textgrid_intervals, build_durations,
    FRAME_RATE, SR,
)

FS2_DIR = ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0"
HIFIGAN_DIR = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"

FEAT_DIR = ROOT / "data" / "unseen_features"
WAV_DIR = ROOT / "output" / "unseen_teacher_wav"
MFA_CORPUS = ROOT / "data" / "unseen_mfa_corpus"
MFA_OUTPUT = ROOT / "data" / "unseen_mfa_aligned"
MANIFEST_PATH = ROOT / "data" / "unseen_manifest.jsonl"
DICT_PATH = ROOT / "data" / "mfa_char_dictionary.txt"

for d in [FEAT_DIR, WAV_DIR, MFA_CORPUS, MFA_OUTPUT]:
    d.mkdir(parents=True, exist_ok=True)

# Truly unseen poems from test_set_v1.json
UNSEEN_POEMS = [
    {
        "poem_id": "unseen_shizhishang",
        "title": "使至塞上",
        "author": "王维",
        "dynasty": "唐",
        "content": "单车欲问边，属国过居延。征蓬出汉塞，归雁入胡天。大漠孤烟直，长河落日圆。萧关逢候骑，都护在燕然。",
    },
    {
        "poem_id": "unseen_jiangjinjiu",
        "title": "将进酒",
        "author": "李白",
        "dynasty": "唐",
        "content": "君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。",
    },
    {
        "poem_id": "unseen_chunjianghuayueye",
        "title": "春江花月夜",
        "author": "张若虚",
        "dynasty": "唐",
        "content": "春江潮水连海平，海上明月共潮生。滟滟随波千万里，何处春��无月明。江流宛转绕芳甸，月照花林皆似霰。空里流霜不觉飞，汀上白沙看不见。",
    },
    {
        "poem_id": "unseen_shudaonan",
        "title": "蜀道难",
        "author": "李白",
        "dynasty": "唐",
        "content": "噫吁嚱，危乎高哉！蜀道之难，难于上青天！蚕丛及鱼凫，开国何茫然。尔来四万八千岁，不与秦塞通人��。西当太白有鸟道，可以横绝峨眉巅。",
    },
    {
        "poem_id": "unseen_pipaxing",
        "title": "琵琶行",
        "author": "白居易",
        "dynasty": "唐",
        "content": "浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。忽闻水上琵琶声，主人忘归客不发。",
    },
    {
        "poem_id": "unseen_shanxing",
        "title": "山行",
        "author": "杜牧",
        "dynasty": "唐",
        "content": "远上寒山石径斜，白云生处有人家。停车坐爱枫林晚，霜叶红于二月花。",
    },
]


def main():
    import onnxruntime as ort
    import soundfile as sf

    print("Loading PaddleSpeech ONNX models...")
    fs2 = ort.InferenceSession(str(FS2_DIR / "fastspeech2_csmsc.onnx"),
                               providers=["CPUExecutionProvider"])
    hifigan = ort.InferenceSession(str(HIFIGAN_DIR / "hifigan_csmsc.onnx"),
                                   providers=["CPUExecutionProvider"])
    phone_map = load_phone_id_map()
    print(f"  FS2 + HiFiGAN loaded. Phone vocab: {len(phone_map)}")

    # Stage 1: Generate teacher audio + features
    print("\n=== Stage 1: Teacher inference + feature extraction ===")
    stage1_results = []

    for poem in UNSEEN_POEMS:
        pid = poem["poem_id"]
        full_text = f"{poem['title']}，{poem['dynasty']}·{poem['author']}。{poem['content']}"
        print(f"\n  [{pid}] {poem['title']}: {full_text[:50]}...")

        # G2P
        phonemes = text_to_phonemes(full_text)
        ids, unmapped = phones_to_ids(phonemes, phone_map)
        if unmapped:
            print(f"    WARN: {len(unmapped)} unmapped: {unmapped[:5]}")

        # FS2 → teacher mel → HiFiGAN → audio
        teacher_mel = run_fs2(fs2, ids)
        audio = run_hifigan(hifigan, teacher_mel)

        # Save teacher wav
        wav_path = str(WAV_DIR / f"{pid}.wav")
        sf.write(wav_path, audio, SR)
        print(f"    Audio: {len(audio)/SR:.1f}s, mel_shape={teacher_mel.shape}")

        # Extract features
        mel = extract_mel(audio, SR)
        f0 = extract_f0(audio, SR)
        energy = extract_energy(mel)

        min_len = min(len(f0), mel.shape[0], len(energy))
        mel = mel[:min_len]
        f0 = f0[:min_len]
        energy = energy[:min_len]

        # Save features
        npz_path = FEAT_DIR / f"{pid}.npz"
        np.savez_compressed(str(npz_path),
                            mel=mel.astype(np.float32),
                            f0=f0.astype(np.float32),
                            energy=energy.astype(np.float32))

        # Save MFA corpus (16kHz)
        import librosa as lr
        audio_16k = lr.resample(audio, orig_sr=SR, target_sr=16000)
        mfa_wav = MFA_CORPUS / f"{pid}.wav"
        sf.write(str(mfa_wav), audio_16k, 16000)
        lab_path = MFA_CORPUS / f"{pid}.lab"
        lab_path.write_text(full_text, encoding="utf-8")

        stage1_results.append({
            "poem_id": pid,
            "title": poem["title"],
            "text": full_text,
            "phoneme_ids": ids,
            "n_phonemes": len(ids),
            "mel_len": min_len,
            "unmapped": unmapped,
        })
        print(f"    Features saved: mel {mel.shape}, f0 {f0.shape}, energy {energy.shape}")

    # Stage 2: MFA alignment (batch)
    print("\n=== Stage 2: MFA batch alignment ===")
    import subprocess
    import shutil

    if MFA_OUTPUT.exists():
        shutil.rmtree(MFA_OUTPUT)
    MFA_OUTPUT.mkdir(parents=True)

    cmd = [
        "mfa", "align",
        str(MFA_CORPUS),
        str(DICT_PATH),
        "mandarin_mfa",
        str(MFA_OUTPUT),
        "--overwrite",
        "--clean",
        "--num_jobs", "4",
        "--beam", "100",
        "--retry_beam", "400",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, timeout=600)
    tg_count = len(list(MFA_OUTPUT.glob("*.TextGrid")))
    print(f"  TextGrids: {tg_count}/{len(UNSEEN_POEMS)}")

    # Stage 3: Build manifest
    print("\n=== Stage 3: Build unseen manifest ===")
    records = []

    for entry in stage1_results:
        pid = entry["poem_id"]
        text = entry["text"]
        ids = entry["phoneme_ids"]
        mel_len = entry["mel_len"]

        tg_path = MFA_OUTPUT / f"{pid}.TextGrid"
        if not tg_path.exists():
            print(f"  SKIP {pid}: no TextGrid")
            continue

        tg_intervals = parse_textgrid_intervals(tg_path)
        phonemes = text_to_phonemes(text)
        durations = build_durations(phonemes, tg_intervals)

        # Fix duration sum to match mel_len
        dur_sum = sum(durations)
        if dur_sum != mel_len:
            diff = mel_len - dur_sum
            max_idx = max(range(len(durations)), key=lambda x: durations[x])
            durations[max_idx] = max(durations[max_idx] + diff, 1)

        if sum(durations) != mel_len:
            print(f"  SKIP {pid}: duration mismatch {sum(durations)} vs {mel_len}")
            continue

        records.append({
            "poem_id": pid,
            "phoneme_ids": ids,
            "durations": durations,
            "mel_path": f"data/unseen_features/{pid}.npz",
            "mel_len": mel_len,
            "n_phonemes": len(ids),
            "text": text,
        })
        print(f"  {pid}: OK (mel_len={mel_len}, n_phonemes={len(ids)})")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nManifest: {MANIFEST_PATH} ({len(records)} records)")
    print("DONE")


if __name__ == "__main__":
    main()
