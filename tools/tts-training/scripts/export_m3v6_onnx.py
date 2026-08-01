"""Export m3_v6 FS2 to ONNX + PyTorch<->ONNX parity on the m3_v6 protocol.

Reuses the proven v9 ONNX wrapper (custom attention, no nn.MultiheadAttention,
dynamic length). m3_v6 has the SAME inference architecture as v9 — only the
training loss weights differed (--w_dursum / --w_dur_linear) and the manifests
were dot-token-fixed. So the wrapper is identical; only the checkpoint, stats
and verification manifests change.

Parity target (torch eval, same manifests):
  train-30 (m3b_train, seed 42): 20.6%   holdout-20 (m3p2b_holdout): 21.0%
"""
import json, sys, math, os, random
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from pathlib import Path

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))

CKPT = ROOT / 'checkpoints' / 'm3_v6_final.pt'
STATS_PATH = ROOT / 'data' / 'm3p2_norm_stats.json'
HOLDOUT_MANIFEST = ROOT / 'data' / 'm3p2b_holdout_manifest.jsonl'
TRAIN_MANIFEST = ROOT / 'data' / 'm3b_train_manifest.jsonl'
OUTDIR = ROOT / 'models' / 'sufei_fs2_onnx_m3v6'
ONNX_PATH = OUTDIR / 'fastspeech2_sufei_m3v6.onnx'


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
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
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

ckpt = torch.load(str(CKPT), map_location='cpu', weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
state_dict = {k: v for k, v in ckpt['model'].items() if not k.startswith('film_gen')}
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded {CKPT.name} (step={ckpt.get('step')})")

wrapper = Fs2OnnxWrapper(model)
OUTDIR.mkdir(parents=True, exist_ok=True)

# torch-side sanity: multiple lengths
for n_phones in [5, 10, 30, 67, 141]:
    dummy = torch.tensor(list(range(n_phones)), dtype=torch.long)
    with torch.no_grad():
        mel = wrapper(dummy)
    print("torch test %d phones -> mel %s" % (n_phones, str(mel.shape)))

# Export
dummy_text = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
torch.onnx.export(
    wrapper, (dummy_text,), str(ONNX_PATH),
    input_names=["text"], output_names=["mel"],
    dynamic_axes={"text": {0: "length"}, "mel": {0: "frames"}},
    opset_version=17,
)
print("\nExported: %s (%.1f MB)" % (ONNX_PATH, os.path.getsize(ONNX_PATH) / 1e6))

# --- ONNX runtime + full E2E parity ---
import onnxruntime as ort
sess = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])

stats = json.load(open(STATS_PATH))
mel_mean = np.array(stats['mel_mean'], dtype=np.float32)
mel_std = np.array(stats['mel_std'], dtype=np.float32)

HIFIGAN = ROOT / 'models' / 'paddlespeech_onnx' / 'hifigan_csmsc_onnx_0.2.0'
voc = ort.InferenceSession(str(HIFIGAN / 'hifigan_csmsc.onnx'), providers=['CPUExecutionProvider'])

import soundfile as sf
from eval_tts import load_asr_model, transcribe
from eval_unified import cer_detail
asr = load_asr_model()


def onnx_e2e_cer(records, prefix):
    cers = []
    for idx, rec in enumerate(records):
        pid = rec['poem_id']
        phone_ids = np.array(rec['phoneme_ids'], dtype=np.int64)
        mel_norm = sess.run(None, {"text": phone_ids})[0]
        mel_raw = mel_norm * mel_std + mel_mean
        audio = voc.run(None, {'logmel': mel_raw.astype(np.float32)})[0].flatten()
        wav = str(OUTDIR / f"{prefix}{pid}.wav")
        sf.write(wav, audio, 24000)
        asr_text, _ = transcribe(asr, wav)
        cer, _, _, _, _ = cer_detail(rec['text'], asr_text)
        cers.append(cer)
        print("  [%2d] %s: CER=%.0f%% | %s" % (idx + 1, pid, cer * 100, asr_text[:40]))
    return np.array(cers)


holdout = [json.loads(l) for l in open(HOLDOUT_MANIFEST, encoding='utf-8')]
train = [json.loads(l) for l in open(TRAIN_MANIFEST, encoding='utf-8')]
random.seed(42)
train_sample = random.sample(train, 30)

print("\n=== ONNX E2E HOLDOUT (20) ===")
hc = onnx_e2e_cer(holdout, 'holdout_')
print("\n=== ONNX E2E TRAIN (30, seed 42) ===")
tc = onnx_e2e_cer(train_sample, 'train_')

print("\n" + "=" * 60)
print("  m3_v6 ONNX vs torch parity")
print("  Train CER:   ONNX %.1f%%  (torch 20.6%%)  gap %+.1fpp" % (tc.mean()*100, tc.mean()*100 - 20.6))
print("  Holdout CER: ONNX %.1f%%  (torch 21.0%%)  gap %+.1fpp" % (hc.mean()*100, hc.mean()*100 - 21.0))
print("=" * 60)

# Copy supporting files for packaging
import shutil
phone_map_src = ROOT / 'models' / 'paddlespeech_onnx' / 'fastspeech2_csmsc_onnx_0.2.0' / 'phone_id_map.txt'
if phone_map_src.exists():
    shutil.copy(phone_map_src, OUTDIR / 'phone_id_map.txt')
np.savez(str(OUTDIR / 'norm_stats.npz'), mel_mean=mel_mean, mel_std=mel_std)

print("\nFiles in %s:" % OUTDIR)
for f in sorted(OUTDIR.iterdir()):
    print("  %s (%.1f KB)" % (f.name, f.stat().st_size / 1024))
print("DONE")
