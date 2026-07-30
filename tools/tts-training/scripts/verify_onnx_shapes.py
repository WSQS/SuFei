"""Verify ONNX model works with variable-length inputs."""
import json, sys, numpy as np, torch
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))

ONNX_PATH = ROOT / 'models' / 'sufei_fs2_onnx' / 'fastspeech2_sufei.onnx'
sess = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])

# Test with different length inputs
train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v3.jsonl', encoding='utf-8')]

for rec in train[:3]:
    pid = rec['poem_id']
    phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)
    text_preview = rec['text'][:20]

    result = sess.run(None, {"text": phone_ids})
    mel = result[0]
    print("%s: %d phones -> mel shape %s, range [%.2f, %.2f] | %s" % (
        pid, len(phone_ids), str(mel.shape), mel.min(), mel.max(), text_preview))

# Check if T_max is baked as constant
print("\nIf all outputs have shape [48, 80], the LengthRegulator T_max is baked as constant (bug).")
print("If shapes vary, dynamic export works correctly.")
