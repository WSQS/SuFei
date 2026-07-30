"""Extract per-phone durations from teacher FS2 ONNX by tracing intermediate outputs."""
import json, sys, numpy as np
import onnxruntime as ort
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))

FS2_ONNX = ROOT / 'models' / 'paddlespeech_onnx' / 'fastspeech2_csmsc_onnx_0.2.0' / 'fastspeech2_csmsc.onnx'

# Create session with all intermediate outputs
so = ort.SessionOptions()
so.inter_op_num_threads = 4

# Load the ONNX model and add the round node as an additional output
import onnx

model_proto = onnx.load(str(FS2_ONNX))

# Find the Round node output name
round_output = None
exp0_output = None
for node in model_proto.graph.node:
    if node.op_type == 'Round':
        round_output = node.output[0]
        print("Found Round output: %s" % round_output)
    if node.name == 'p2o.Exp.0' and node.op_type == 'Exp':
        exp0_output = node.output[0]
        print("Found Exp.0 output: %s" % exp0_output)

# Also find the Exp before Round to get log-durations
# The graph does: predictor output -> Exp -> Round
# Let's also grab the Clip output (before Exp)

# Add round_output as a graph output
from onnx import helper, ValueInfoProto

# Create a new output spec
new_output = helper.make_tensor_value_info(round_output, onnx.TensorProto.FLOAT, [-1, -1])
model_proto.graph.output.append(new_output)

if exp0_output:
    new_output2 = helper.make_tensor_value_info(exp0_output, onnx.TensorProto.FLOAT, [-1, -1])
    model_proto.graph.output.append(new_output2)

# Save modified model
modified_path = str(ROOT / 'models' / 'paddlespeech_onnx' / 'fastspeech2_csmsc_with_dur.onnx')
onnx.save(model_proto, modified_path)
print("Saved modified ONNX with duration output: %s" % modified_path)

# Test it
sess = ort.InferenceSession(modified_path, providers=['CPUExecutionProvider'])

train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v2.jsonl', encoding='utf-8')]

# Test on 5 samples
for rec in train[:5]:
    pid = rec['poem_id']
    phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)
    outputs = sess.run(None, {"text": phone_ids})

    mel = outputs[0]
    if mel.ndim == 3: mel = mel[0]

    # Find which output is the duration (round output is integer-like)
    for i in range(1, len(outputs)):
        out = outputs[i]
        if out.ndim == 2:
            dur_raw = out[0]  # remove batch dim
            dur_int = dur_raw.astype(int)
            print("\n%s: %d phonemes" % (pid, len(phone_ids)))
            print("  output[%d] shape=%s, sum=%d, mel_frames=%d" % (
                i, out.shape, dur_int.sum(), mel.shape[0]))
            print("  dur[:10]=%s" % str(dur_int[:10].tolist()))
            print("  ratio sum(dur)/mel_frames = %.3f" % (dur_int.sum() / mel.shape[0]))
            if abs(dur_int.sum() - mel.shape[0]) <= 2:
                print("  ** MATCH! This is the teacher duration output")

print("\nDone! Now run full extraction on all 320 samples.")
