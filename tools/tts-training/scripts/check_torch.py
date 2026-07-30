import torch
print("torch version:", torch.__version__)
print("has dynamo_export:", hasattr(torch.onnx, 'dynamo_export'))
