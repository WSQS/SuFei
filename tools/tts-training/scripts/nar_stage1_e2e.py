"""Stage 1: End-to-end ONNX inference — Chinese poem → speech."""
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FS2_DIR = ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0"
HIFIGAN_DIR = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
OUTPUT_DIR = ROOT / "output" / "nar_probe"


def load_phone_id_map():
    """Load phone -> id mapping."""
    phone_map = {}
    with open(FS2_DIR / "phone_id_map.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                phone_map[parts[0]] = int(parts[1])
    return phone_map


# PaddleSpeech initials (声母)
INITIALS = {"b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x",
            "zh", "ch", "sh", "r", "z", "c", "s"}

# pypinyin format → PaddleSpeech format mapping
# PaddleSpeech uses: i for j/q/x, ii for z/c/s, iii for zh/ch/sh/r
# Also: uen for un(wen), ve for üe, iou for you, etc.
FINAL_MAP = {
    # Standard pinyin final → PaddleSpeech final
    # i-variants (will be disambiguated by initial later)
    "i": "i",
    "ia": "ia", "ian": "ian", "iang": "iang", "iao": "iao",
    "ie": "ie", "in": "in", "ing": "ing", "iong": "iong",
    "iou": "iou", "io": "io",
    # u-variants
    "u": "u", "ua": "ua", "uai": "uai", "uan": "uan",
    "uang": "uang", "uei": "uei", "uen": "uen", "ueng": "ueng",
    "uo": "uo", "ou": "ou",
    # ü-variants (pypinyin uses v or ü)
    "v": "v", "van": "van", "ve": "ve", "vn": "vn",
    # a-variants
    "a": "a", "ai": "ai", "an": "an", "ang": "ang", "ao": "ao",
    # e-variants
    "e": "e", "ei": "ei", "en": "en", "eng": "eng", "er": "er",
    # o-variants
    "o": "o", "ong": "ong",
}

# pypinyin uses some non-standard finals that need remapping
PYPINYIN_FINAL_FIX = {
    "un": "uen",        # 春 chun → ch uen
    "ui": "uei",        # 水 shui → sh uei
    "iu": "iou",        # 秋 qiu → q iou
    "ue": "ve",         # 觉 jue → j ve
    "ü": "v",           # 女 nü → n v
    "üe": "ve",         # 月 yue → ve (no initial)
    "n": "en",          # 嗯 → en (edge case)
}


def text_to_phones(text):
    """Convert Chinese text to PaddleSpeech-style phoneme sequence."""
    from pypinyin import pinyin, Style

    phones = []
    for char in text:
        # Handle punctuation
        if char in "，":
            phones.append("，")
            continue
        elif char in "。":
            phones.append("。")
            continue
        elif char in "？":
            phones.append("？")
            continue
        elif char in "！":
            phones.append("！")
            continue

        py_list = pinyin(char, style=Style.TONE3, errors="default")
        if not py_list or not py_list[0]:
            continue
        py = py_list[0][0]

        if not py or py == char:
            continue

        # Extract tone
        tone = ""
        if py[-1].isdigit():
            tone = py[-1]
            py_base = py[:-1]
        else:
            # No tone = neutral tone (轻声), use 5
            tone = "5"
            py_base = py

        # Handle y/w pseudo-initials: convert to proper finals
        if py_base[0] == "y":
            if len(py_base) == 1:
                py_base = "i"  # just "yi"
            elif py_base == "you":
                py_base = "iou"
            elif py_base == "yue":
                py_base = "ve"
            elif py_base == "yu":
                py_base = "v"
            elif py_base == "yuan":
                py_base = "van"
            elif py_base == "yun":
                py_base = "vn"
            elif py_base == "yong":
                py_base = "iong"
            elif py_base[1] in "ae":
                py_base = "i" + py_base[1:]  # ya→ia, yan→ian, yang→iang, yao→iao
            elif py_base[1] == "i":
                py_base = py_base[1:]  # yii→ii edge case
        elif py_base[0] == "w":
            if len(py_base) == 1:
                py_base = "u"  # just "wu"
            elif py_base == "wu":
                py_base = "u"
            else:
                py_base = "u" + py_base[1:]  # wa→ua, wan→uan, wang→uang, wei→uei, wen→uen, weng→ueng

        # Fix pypinyin-specific finals
        for old, new in PYPINYIN_FINAL_FIX.items():
            if py_base == old:
                py_base = new
                break

        # Try to split into initial + final
        matched = False
        for init_len in [2, 1]:
            if len(py_base) > init_len:
                candidate_init = py_base[:init_len]
                candidate_final = py_base[init_len:]
                if candidate_init in INITIALS:
                    # Disambiguate i → i/ii/iii based on initial
                    if candidate_final.rstrip("12345").rstrip("r") == "i":
                        if candidate_init in {"j", "q", "x"}:
                            pass  # stays "i"
                        elif candidate_init in {"z", "c", "s"}:
                            candidate_final = candidate_final.replace("i", "ii")
                        elif candidate_init in {"zh", "ch", "sh", "r"}:
                            candidate_final = candidate_final.replace("i", "iii")

                    # Fix final if needed
                    final_base = candidate_final.rstrip("12345").rstrip("r")
                    if final_base in PYPINYIN_FINAL_FIX:
                        new_base = PYPINYIN_FINAL_FIX[final_base]
                        tone_part = candidate_final[len(final_base):]
                        candidate_final = new_base + tone_part

                    phones.append(candidate_init)
                    phones.append(candidate_final + tone)
                    matched = True
                    break

        if not matched:
            # No initial — fix standalone finals
            for old, new in PYPINYIN_FINAL_FIX.items():
                if py_base == old:
                    py_base = new
                    break
            phones.append(py_base + tone)

    # Add EOS
    phones.append("<eos>")
    return phones


def phones_to_ids(phones, phone_map):
    """Convert phone list to ID list."""
    ids = []
    for p in phones:
        if p in phone_map:
            ids.append(phone_map[p])
        else:
            print(f"  WARN: phone '{p}' not in vocab, skipping")
    return ids


def main():
    phone_map = load_phone_id_map()
    print(f"Phone vocab: {len(phone_map)} entries")

    # Load ONNX models
    print("Loading FastSpeech2 ONNX...")
    fs2_sess = ort.InferenceSession(str(FS2_DIR / "fastspeech2_csmsc.onnx"))

    print("Loading HiFiGAN ONNX...")
    hifigan_sess = ort.InferenceSession(str(HIFIGAN_DIR / "hifigan_csmsc.onnx"))

    # Test poems
    TEST_TEXTS = [
        ("chunxiao", "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"),
        ("jingyesi", "床前明月光，疑是地上霜。��头望明月，低头思故乡。"),
        ("wangyue", "岱宗夫如何？齐鲁青未了。造化钟神秀，阴阳割昏晓。"),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, text in TEST_TEXTS:
        print(f"\n--- {name}: {text[:20]}... ---")

        # G2P
        phones = text_to_phones(text)
        print(f"  Phones: {' '.join(phones[:20])}... ({len(phones)} total)")

        phone_ids = phones_to_ids(phones, phone_map)
        print(f"  IDs: {phone_ids[:10]}... ({len(phone_ids)} total)")

        if not phone_ids:
            print("  ERROR: no valid phone IDs")
            continue

        # FastSpeech2 inference
        t0 = time.time()
        text_arr = np.array(phone_ids, dtype=np.int64)
        mel = fs2_sess.run(None, {"text": text_arr})[0]
        fs2_time = time.time() - t0
        print(f"  FS2: mel shape={mel.shape}, {fs2_time:.3f}s")

        # HiFiGAN inference
        t0 = time.time()
        audio = hifigan_sess.run(None, {"logmel": mel.astype(np.float32)})[0]
        voc_time = time.time() - t0
        print(f"  HiFiGAN: audio shape={audio.shape}, {voc_time:.3f}s")

        # Save WAV
        audio_data = audio.flatten()
        out_path = OUTPUT_DIR / f"{name}_nar.wav"
        sf.write(str(out_path), audio_data, 24000)
        duration = len(audio_data) / 24000
        print(f"  Saved: {out_path.name} ({duration:.1f}s, {out_path.stat().st_size/1024:.0f}KB)")

    print(f"\n=== Stage 1 end-to-end inference complete ===")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
