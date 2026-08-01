"""Package m3_v6 config-B (both fp16) into app-expected filenames + integrity manifest.

Produces models/sufei_m3v6_deploy/ with the exact names NarOnnxEngine expects:
  fastspeech2_sufei.onnx  hifigan_csmsc.onnx  phone_id_map.txt  norm_stats.npz
plus SHA256SUMS.txt and manifest.json (model version, quantization, sha256, sizes,
eval metrics, G2P convention) for HF pinned-revision hosting + on-device integrity.
"""
import hashlib, json, shutil
from pathlib import Path

ROOT = Path('E:/sufei-training')
SRC = ROOT / 'models' / 'sufei_fs2_onnx_m3v6'
DEPLOY = ROOT / 'models' / 'sufei_m3v6_deploy'
DEPLOY.mkdir(parents=True, exist_ok=True)

# (src name in SRC, app-expected deploy name)
FILES = [
    ('fastspeech2_sufei_m3v6_fp16.onnx', 'fastspeech2_sufei.onnx'),
    ('hifigan_csmsc_fp16.onnx', 'hifigan_csmsc.onnx'),
    ('phone_id_map.txt', 'phone_id_map.txt'),
    ('norm_stats.npz', 'norm_stats.npz'),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


entries = []
sums_lines = []
for src_name, dst_name in FILES:
    src = SRC / src_name
    dst = DEPLOY / dst_name
    shutil.copyfile(src, dst)
    digest = sha256(dst)
    size = dst.stat().st_size
    entries.append({'file': dst_name, 'sha256': digest, 'bytes': size})
    sums_lines.append(f"{digest}  {dst_name}")
    print(f"{dst_name:26s} {size/1e6:7.2f} MB  {digest[:16]}...")

total = sum(e['bytes'] for e in entries)
manifest = {
    'model': 'sufei_fs2_m3v6',
    'description': 'SuFei Chinese-poem TTS: FastSpeech2 (7.6M) + HiFi-GAN, fp16',
    'quantization': 'fp16 (weights+activations; length-regulator sequence ops + '
                    'norm/range ops kept fp32; I/O fp32 via keep_io_types)',
    'sample_rate': 24000,
    'mel_bins': 80,
    'phone_vocab': 268,
    'checkpoint': 'm3_v6_final.pt (step 30000)',
    'eval': {'protocol': 'holdout-20 / train-30 seed42, ASR-CER (FunASR)',
             'train_cer': 20.3, 'holdout_cer': 21.9,
             'fp32_baseline': {'train_cer': 20.6, 'holdout_cer': 21.0},
             'production_v9': {'train_cer': 24.0, 'holdout_cer': 21.5}},
    'g2p_convention': "PaddleSpeech phones; '·' produces NO token (dot-token fix, "
                      "m3_v6); punct map ，。？！ kept, ；：->，",
    'pipeline': 'text -> G2P -> phone_ids -> fastspeech2_sufei.onnx -> mel[T,80] '
                '-> *mel_std+mel_mean (norm_stats.npz) -> hifigan_csmsc.onnx -> 24k pcm',
    'total_bytes': total,
    'files': entries,
}
(DEPLOY / 'SHA256SUMS.txt').write_text('\n'.join(sums_lines) + '\n', encoding='utf-8')
(DEPLOY / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                      encoding='utf-8')

print(f"\nTotal shipping size: {total/1e6:.1f} MB")
print(f"Deploy dir: {DEPLOY}")
for f in sorted(DEPLOY.iterdir()):
    print(f"  {f.name} ({f.stat().st_size} B)")
print("DONE")
