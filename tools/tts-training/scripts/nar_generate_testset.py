"""Generate 24 test poems with PaddleSpeech FS2 ONNX baseline."""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nar_stage1_e2e import text_to_phones, load_phone_id_map, phones_to_ids

ROOT = Path(__file__).resolve().parents[1]
FS2_DIR = ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0"
HIFIGAN_DIR = ROOT / "models" / "paddlespeech_onnx" / "hifigan_csmsc_onnx_0.2.0"
TEST_SET = ROOT / "data" / "test_set_v1.json"
OUTPUT_DIR = ROOT / "output" / "nar_baseline"


def main():
    phone_map = load_phone_id_map()
    print(f"Phone vocab: {len(phone_map)}")

    print("Loading FastSpeech2 ONNX...")
    fs2_sess = ort.InferenceSession(str(FS2_DIR / "fastspeech2_csmsc.onnx"))
    print("Loading HiFiGAN ONNX...")
    hifigan_sess = ort.InferenceSession(str(HIFIGAN_DIR / "hifigan_csmsc.onnx"))

    with open(TEST_SET, encoding="utf-8") as f:
        test_set = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for s in test_set["samples"]:
        pid = s["poem_id"]
        title = s["title"]
        content = s["content"]
        # Use content with original punctuation for G2P
        text = content

        print(f"[{pid}] {title}...", end=" ", flush=True)

        phones = text_to_phones(text)
        phone_ids = phones_to_ids(phones, phone_map)

        if not phone_ids:
            print("ERROR: no valid phone IDs")
            continue

        t0 = time.time()
        text_arr = np.array(phone_ids, dtype=np.int64)
        mel = fs2_sess.run(None, {"text": text_arr})[0]
        audio = hifigan_sess.run(None, {"logmel": mel.astype(np.float32)})[0]
        audio_data = audio.flatten()

        out_path = OUTPUT_DIR / f"{pid}_nar_baseline.wav"
        sf.write(str(out_path), audio_data, 24000)
        duration = len(audio_data) / 24000
        print(f"{duration:.1f}s, {len(phones)} phones")

    print(f"\nDone: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
