"""Export v9 FS2 to ONNX with custom attention (no nn.MultiheadAttention)."""
import json, sys, math, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))


class OnnxAttention(nn.Module):
    """ONNX-friendly multi-head attention that doesn't bake sequence length."""
    def __init__(self, d_model=256, nhead=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L, L]
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, L, D]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class OnnxTransformerBlock(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.attn = OnnxAttention(original.self_attn.embed_dim, original.self_attn.num_heads)
        self.norm1 = original.norm1
        self.norm2 = original.norm2
        self.linear1 = original.linear1
        self.linear2 = original.linear2

        # Copy weights from original MultiheadAttention
        with torch.no_grad():
            mha = original.self_attn
            self.attn.q_proj.weight.copy_(mha.in_proj_weight[:mha.embed_dim])
            self.attn.q_proj.bias.copy_(mha.in_proj_bias[:mha.embed_dim])
            self.attn.k_proj.weight.copy_(mha.in_proj_weight[mha.embed_dim:2*mha.embed_dim])
            self.attn.k_proj.bias.copy_(mha.in_proj_bias[mha.embed_dim:2*mha.embed_dim])
            self.attn.v_proj.weight.copy_(mha.in_proj_weight[2*mha.embed_dim:])
            self.attn.v_proj.bias.copy_(mha.in_proj_bias[2*mha.embed_dim:])
            self.attn.out_proj.weight.copy_(mha.out_proj.weight)
            self.attn.out_proj.bias.copy_(mha.out_proj.bias)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.linear2(F.relu(self.linear1(x))))
        return x


class LengthRegulatorOnnx(nn.Module):
    def forward(self, x, durations):
        dur_flat = durations.squeeze(0)
        if x.dim() == 3:
            return torch.repeat_interleave(x, dur_flat, dim=1)
        else:
            return torch.repeat_interleave(x, dur_flat, dim=0)


class Fs2OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embedding = model.embedding
        self.pos_enc = model.pos_enc
        self.encoder_layers = nn.ModuleList([OnnxTransformerBlock(l) for l in model.encoder_layers])
        self.duration_predictor = model.duration_predictor
        self.pitch_predictor = model.pitch_predictor
        self.energy_predictor = model.energy_predictor
        self.length_regulator = LengthRegulatorOnnx()
        self.pitch_embed = model.pitch_embed
        self.energy_embed = model.energy_embed
        self.decoder_layers = nn.ModuleList([OnnxTransformerBlock(l) for l in model.decoder_layers])
        self.mel_linear = model.mel_linear
        self.d_model = model.d_model

    def forward(self, text):
        x = self.embedding(text) * math.sqrt(self.d_model)
        x = self.pos_enc(x.unsqueeze(0))
        for layer in self.encoder_layers:
            x = layer(x)

        log_dur = self.duration_predictor(x)
        dur = log_dur.exp().round().clamp(min=1).long()

        mel_input = self.length_regulator(x, dur)

        pitch_pred = self.pitch_predictor(x)
        energy_pred = self.energy_predictor(x)

        pitch_exp = self.length_regulator(pitch_pred.unsqueeze(-1), dur)
        energy_exp = self.length_regulator(energy_pred.unsqueeze(-1), dur)

        mel_input = mel_input + self.pitch_embed(pitch_exp) + self.energy_embed(energy_exp)
        mel_input = self.pos_enc(mel_input)

        dec = mel_input
        for layer in self.decoder_layers:
            dec = layer(dec)
        mel_out = self.mel_linear(dec)

        return mel_out[0]


from nar_fastspeech2 import FastSpeech2

ckpt = torch.load(str(ROOT / 'checkpoints' / 'fs2_final.pt'), map_location='cpu', weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
state_dict = {k: v for k, v in ckpt['model'].items() if not k.startswith('film_gen')}
model.load_state_dict(state_dict)
model.eval()

wrapper = Fs2OnnxWrapper(model)

OUTDIR = ROOT / 'models' / 'sufei_fs2_onnx'
OUTDIR.mkdir(parents=True, exist_ok=True)

# Test with multiple lengths
for n_phones in [5, 10, 30, 67, 141]:
    dummy = torch.tensor(list(range(n_phones)), dtype=torch.long)
    with torch.no_grad():
        mel = wrapper(dummy)
    print("torch test %d phones -> mel %s" % (n_phones, str(mel.shape)))

# Export
dummy_text = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
onnx_path = str(OUTDIR / 'fastspeech2_sufei.onnx')

torch.onnx.export(
    wrapper,
    (dummy_text,),
    onnx_path,
    input_names=["text"],
    output_names=["mel"],
    dynamic_axes={
        "text": {0: "length"},
        "mel": {0: "frames"},
    },
    opset_version=17,
)

print("\nExported: %s" % onnx_path)
import os
size_mb = os.path.getsize(onnx_path) / 1e6
print("Model size: %.1f MB" % size_mb)

# Verify with ONNX runtime — test variable lengths
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

train = [json.loads(l) for l in open(ROOT / 'data/train_300_manifest_v3.jsonl', encoding='utf-8')]
print("\nONNX verification:")
for rec in train[:5]:
    pid = rec['poem_id']
    phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)
    result = sess.run(None, {"text": phone_ids})
    mel = result[0]
    print("  %s: %d phones -> mel %s, range [%.2f, %.2f]" % (
        pid, len(phone_ids), str(mel.shape), mel.min(), mel.max()))

# Copy supporting files
import shutil
phone_map_src = ROOT / 'models' / 'paddlespeech_onnx' / 'fastspeech2_csmsc_onnx_0.2.0' / 'phone_id_map.txt'
if phone_map_src.exists():
    shutil.copy(phone_map_src, OUTDIR / 'phone_id_map.txt')

stats = json.load(open(ROOT / 'data' / 'train_300_norm_stats_v3.json'))
np.savez(
    str(OUTDIR / 'norm_stats.npz'),
    mel_mean=np.array(stats['mel_mean'], dtype=np.float32),
    mel_std=np.array(stats['mel_std'], dtype=np.float32),
)

# Full end-to-end: ONNX FS2 -> denorm -> HiFiGAN -> audio
print("\n=== Full E2E test (ONNX FS2 + HiFiGAN) ===")
HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0'
voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'), providers=['CPUExecutionProvider'])

mel_mean = np.array(stats['mel_mean'], dtype=np.float32)
mel_std = np.array(stats['mel_std'], dtype=np.float32)

import soundfile as sf
sys.path.insert(0, str(ROOT / 'scripts'))
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail

asr = load_asr_model()
cers = []
for rec in train[:10]:
    pid = rec['poem_id']
    phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)
    text = rec['text']
    mel_norm = sess.run(None, {"text": phone_ids})[0]
    mel_raw = mel_norm * mel_std + mel_mean
    audio = voc.run(None, {'logmel': mel_raw.astype(np.float32)})[0].flatten()
    wav = str(OUTDIR / f"test_{pid}.wav")
    sf.write(wav, audio, 24000)
    asr_text, _ = transcribe(asr, wav)
    cer, _, _, _, _ = cer_detail(text, asr_text)
    cers.append(cer)
    print("  %s: CER=%.0f%% | %s" % (pid, cer*100, asr_text[:30]))

print("\nONNX E2E mean CER: %.1f%% (torch eval was 24.0%%)" % (np.mean(cers)*100))
print("\nFiles:")
for f in OUTDIR.iterdir():
    print("  %s (%.1f KB)" % (f.name, f.stat().st_size / 1024))
