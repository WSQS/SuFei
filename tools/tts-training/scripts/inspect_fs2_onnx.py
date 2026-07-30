"""Inspect PaddleSpeech FS2 ONNX graph for duration-related nodes."""
import onnxruntime as ort
import onnx
from pathlib import Path

ROOT = Path('E:/sufei-training')
FS2_ONNX = ROOT / 'models' / 'paddlespeech_onnx' / 'fastspeech2_csmsc_onnx_0.2.0' / 'fastspeech2_csmsc.onnx'

# Load ONNX model to inspect graph
model = onnx.load(str(FS2_ONNX))
print("=== Inputs ===")
for inp in model.graph.input:
    print("  %s: %s" % (inp.name, [d.dim_value for d in inp.type.tensor_type.shape.dim]))

print("\n=== Outputs ===")
for out in model.graph.output:
    print("  %s: %s" % (out.name, [d.dim_value for d in out.type.tensor_type.shape.dim]))

print("\n=== All node names (looking for duration/pitch/energy) ===")
for node in model.graph.node:
    name = node.name if node.name else "(unnamed)"
    op = node.op_type
    if any(kw in name.lower() for kw in ['dur', 'pitch', 'energy', 'predictor', 'length', 'regulat', 'expand']):
        print("  ** %s [%s] inputs=%s outputs=%s" % (name, op, list(node.input), list(node.output)))

print("\n=== Node count by op_type ===")
from collections import Counter
ops = Counter(node.op_type for node in model.graph.node)
for op, count in ops.most_common():
    print("  %s: %d" % (op, count))

print("\n=== All node names containing 'Exp' or 'Round' or 'Cast' ===")
for node in model.graph.node:
    name = node.name if node.name else "(unnamed)"
    if node.op_type in ['Exp', 'Round', 'Cast', 'CumSum', 'Gather', 'ReduceSum']:
        print("  %s [%s] -> %s" % (name, node.op_type, list(node.output)[:2]))
