@echo off
C:\Users\wehao\anaconda3\envs\sufei-tts\python.exe -c "import json,statistics; lines=open(r'E:/sufei-training/data/train_300_manifest.jsonl',encoding='utf-8').readlines(); durs=[d for l in lines for d in json.loads(l)['durations']]; print(f'n={len(lines)}, dur median={statistics.median(durs):.1f}, dur=2 frac={sum(1 for d in durs if d==2)/len(durs):.1%%}')"
