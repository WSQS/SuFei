"""Stage 3: Build unseen manifest from TextGrids + features."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
sys.path.insert(0, str(ROOT / "scripts"))

from generate_paddlespeech_distillation_data import (
    text_to_phonemes, phones_to_ids, load_phone_id_map,
    parse_textgrid_intervals, build_durations,
    FRAME_RATE,
)

FEAT_DIR = ROOT / "data" / "unseen_features"
MFA_OUTPUT = ROOT / "data" / "unseen_mfa_aligned"
MANIFEST_PATH = ROOT / "data" / "unseen_manifest.jsonl"

UNSEEN_POEMS = [
    {"poem_id": "unseen_shizhishang", "title": "使至塞上", "author": "王维", "dynasty": "唐",
     "content": "单车欲问边，属国过居延。征蓬出汉塞，归雁入胡天。大漠孤烟直，长河落日圆。萧关逢候骑，都护在燕然。"},
    {"poem_id": "unseen_jiangjinjiu", "title": "将进酒", "author": "李白", "dynasty": "唐",
     "content": "君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。"},
    {"poem_id": "unseen_chunjianghuayueye", "title": "春江花月夜", "author": "张若虚", "dynasty": "唐",
     "content": "春江潮水连海平，海上明月共潮生。滟滟随波千万里，何处春江无月明。江流宛转绕芳甸，月照花林皆似霰。空里流霜不觉飞，汀上白沙看不见。"},
    {"poem_id": "unseen_shudaonan", "title": "蜀道难", "author": "李白", "dynasty": "唐",
     "content": "噫吁嚱，危乎高哉！蜀道之难，难于上青天！蚕丛及鱼凫，开国何茫然。尔来四万八千岁，不与秦塞通人烟。西当太白有鸟道，可以横绝峨眉巅。"},
    {"poem_id": "unseen_pipaxing", "title": "琵琶行", "author": "白居易", "dynasty": "唐",
     "content": "浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。忽闻水���琵琶声，主人忘归客不发。"},
    {"poem_id": "unseen_shanxing", "title": "山行", "author": "杜牧", "dynasty": "唐",
     "content": "远上寒山石径斜，白云生处有人家。停车坐爱枫林晚，霜叶红于二月花。"},
]

phone_map = load_phone_id_map()

records = []
for poem in UNSEEN_POEMS:
    pid = poem["poem_id"]
    full_text = f"{poem['title']}，{poem['dynasty']}·{poem['author']}。{poem['content']}"

    npz = np.load(str(FEAT_DIR / f"{pid}.npz"))
    mel_len = npz["mel"].shape[0]

    tg_path = MFA_OUTPUT / f"{pid}.TextGrid"
    if not tg_path.exists():
        print(f"SKIP {pid}: no TextGrid")
        continue

    tg_intervals = parse_textgrid_intervals(tg_path)
    phonemes = text_to_phonemes(full_text)
    ids, unmapped = phones_to_ids(phonemes, phone_map)
    durations = build_durations(phonemes, tg_intervals)

    dur_sum = sum(durations)
    if dur_sum != mel_len:
        diff = mel_len - dur_sum
        max_idx = max(range(len(durations)), key=lambda x: durations[x])
        durations[max_idx] = max(durations[max_idx] + diff, 1)

    final_sum = sum(durations)
    if final_sum != mel_len:
        print(f"SKIP {pid}: dur mismatch {final_sum} vs {mel_len}")
        continue

    records.append({
        "poem_id": pid,
        "phoneme_ids": ids,
        "durations": durations,
        "mel_path": f"data/unseen_features/{pid}.npz",
        "mel_len": mel_len,
        "n_phonemes": len(ids),
        "text": full_text,
    })
    print(f"  {pid}: OK mel_len={mel_len} n_phonemes={len(ids)} dur_sum={final_sum}")

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nManifest: {MANIFEST_PATH} ({len(records)} records)")
