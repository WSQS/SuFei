"""Diag: per-phone inference durations for poem_0027, two checkpoints.

Usage: diag_dur_0027.py [ckptA ckptB [manifestA manifestB]]
  defaults: m3_v4_final.pt m3_v5_final.pt, both on data/m3_train_manifest.jsonl

Each checkpoint reads poem_0027's phoneme_ids from its own manifest (m3_v6
trained on dot-token-fixed manifests has a shorter sequence). Same-length
sequences print side-by-side; different lengths print two blocks.

MAS training alignments are not stored in checkpoints, so this covers the
inference side only.
"""
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path('E:/sufei-training')
sys.path.insert(0, str(ROOT / 'scripts'))
from nar_fastspeech2 import FastSpeech2

FS2_LOCAL = ROOT / 'models' / 'paddlespeech_onnx' / 'fastspeech2_csmsc_onnx_0.2.0'
id2ph = {}
for line in open(FS2_LOCAL / 'phone_id_map.txt', encoding='utf-8'):
    ph, i = line.split()
    id2ph[int(i)] = ph

CKPTS = sys.argv[1:3] if len(sys.argv) >= 3 else ['m3_v4_final.pt', 'm3_v5_final.pt']
MANIFESTS = sys.argv[3:5] if len(sys.argv) >= 5 else ['data/m3_train_manifest.jsonl'] * 2


def load_rec(manifest):
    for line in open(ROOT / manifest, encoding='utf-8'):
        r = json.loads(line)
        if r['poem_id'] == 'poem_0027':
            return r
    raise SystemExit(f"poem_0027 not in {manifest}")


def infer_dur(ckpt_name, ids):
    ckpt = torch.load(str(ROOT / 'checkpoints' / ckpt_name), map_location='cpu',
                      weights_only=False)
    model = FastSpeech2(vocab_size=268, dropout=0.0)
    sd = {k: v for k, v in ckpt['model'].items() if not k.startswith('film_gen')}
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        pi = torch.tensor([ids], dtype=torch.long)
        x = model.embedding(pi) * math.sqrt(model.d_model)
        x = model.pos_enc(x)
        for layer in model.encoder_layers:
            x = layer(x)
        log_dur = model.duration_predictor(x)
        return log_dur.exp().squeeze(0)


out = open(ROOT / 'logs' / 'diag_dur_0027.txt', 'w', encoding='utf-8')
recs = [load_rec(m) for m in MANIFESTS]
durs = []
for name, rec in zip(CKPTS, recs):
    d = infer_dur(name, rec['phoneme_ids'])
    durs.append(d)
    out.write(f"{name}: total={d.sum().item():.0f} frames "
              f"(mel_len target {rec['mel_len']}, n_tok {len(rec['phoneme_ids'])})\n")

names = [c.replace('_final.pt', '') for c in CKPTS]
if len(recs[0]['phoneme_ids']) == len(recs[1]['phoneme_ids']):
    out.write(f"\nidx phone    {names[0]:>8} {names[1]:>8}   (frames, 12.5ms each)\n")
    for i, pid in enumerate(recs[0]['phoneme_ids']):
        mark = '  <<<' if 13 <= i <= 23 else ''
        out.write(f"{i:3d} {id2ph[pid]:<8} {durs[0][i].item():8.1f} "
                  f"{durs[1][i].item():8.1f}{mark}\n")
else:
    for name, rec, d in zip(names, recs, durs):
        out.write(f"\n--- {name} ({MANIFESTS[names.index(name)]}) ---\n")
        for i, pid in enumerate(rec['phoneme_ids']):
            mark = '  <<<' if 13 <= i <= 23 else ''
            out.write(f"{i:3d} {id2ph[pid]:<8} {d[i].item():8.1f}{mark}\n")
out.close()
print("DONE")
