#!/usr/bin/env python
"""Generate audio for 古诗十九首 (exact originals only)."""
import sys, os, json, re, time
from pathlib import Path

COSYVOICE_DIR = Path.home() / "CosyVoice"
sys.path.insert(0, str(COSYVOICE_DIR))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

OUTPUT_DIR = COSYVOICE_DIR / "data" / "raw_audio_19"
POEMS_DIR = Path("/mnt/c/Users/wsqsy/Documents/android/SuFei/app/src/main/assets")
MODEL_DIR = str(COSYVOICE_DIR / "pretrained_models" / "Fun-CosyVoice3-0.5B")
SAMPLE_RATE = 24000

# Exact source URLs for the 19 original poems
TARGET_URLS = {
    "cfa983aabd96",  # 行行重行行
    "fb281c04a23e",  # 青青河畔草
    "2a844a2e6460",  # 青青陵上柏
    "55c21cc36e9c",  # 今日良宴会
    "97ba2b134232",  # 西北有高楼
    "c72d94b49cc6",  # 涉江采芙蓉
    "d15a88583601",  # 明月皎夜光
    "2ef2ab14a254",  # 冉冉孤生竹
    "d4f03889eaad",  # 庭中有奇树
    "7f09a756c9c0",  # 迢迢牵牛星
    "7969b037360b",  # 回车驾言迈
    "0046a6e4e3e8",  # 东城高且长
    "1b00492cdb81",  # 驱车上东门
    "566eb225e53e",  # 去者日以疏
    "4dedd807c808",  # 生年不满百
    "74f93a1d0f88",  # 凛凛岁云暮
    "2c7d1e6ffc71",  # 孟冬寒气至
    "6105b29267b5",  # 客从远方来
    "a7e1ebce8d01",  # 明月何皎皎
}

INSTRUCTION = "使用自然、克制的中文诗歌朗读方式。声音沉静而清晰。"
PROMPT_WAV = str(COSYVOICE_DIR / "asset" / "zero_shot_prompt.wav")

# 多音字修正表：{诗序号: [(原文片段, 替换为)]}
# 使用 CosyVoice3 的拼音标注格式：在字后面用 [拼音] 覆盖
PROSODY_FIXES = {
    14: [  # 行行重行行
        ("行行重行行", "行行[chóng]重行行"),
        ("会面安可知", "会面[ān]可知"),  # 安 ān
        ("越鸟巢南枝", "越鸟巢南[zhī]"),  # 枝 zhī
        ("游子不顾反", "游子不顾反[fǎn]"),  # 反 fǎn
        ("弃捐勿复道", "弃捐勿复道[dào]"),  # 道 dào
    ],
}


def apply_prosody_fixes(text, poem_index):
    """Apply pinyin annotation for polyphonic characters."""
    fixes = PROSODY_FIXES.get(poem_index, [])
    for original, fixed in fixes:
        text = text.replace(original, fixed)
    return text

def clean_text(text):
    text = text.replace("\n", "")
    text = re.sub(r"[（(][^)）]*[)）]", "", text)
    return text.strip()

def load_19_poems():
    poems = []
    for f in sorted(POEMS_DIR.glob("poems_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                p = json.loads(line)
                url = p.get("sourceUrl", "")
                for hid in TARGET_URLS:
                    if hid in url:
                        poems.append(p)
                        break
    poems.sort(key=lambda p: list(TARGET_URLS).index(
        next(h for h in TARGET_URLS if h in p.get("sourceUrl", ""))
    ))
    return poems

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CosyVoice3...")
    model = AutoModel(model_dir=MODEL_DIR)

    poems = load_19_poems()
    print(f"Found {len(poems)}/19 古诗十九首")

    ok = 0
    for i, poem in enumerate(poems):
        title = poem.get("title", "?")
        content = clean_text(poem["content"])
        author = poem.get("author", "")
        dynasty = poem.get("dynasty", "")

        # Build text with title prefix
        prefix = f"{title}，{dynasty}·{author}。" if dynasty and author else f"{title}。"
        full_text = prefix + content

        # Apply polyphonic character fixes (poem index is 1-based)
        full_text = apply_prosody_fixes(full_text, i + 1)

        url = poem.get("sourceUrl", "")
        hid = next((h for h in TARGET_URLS if h in url), f"poem_{i}")
        out_path = OUTPUT_DIR / f"{i+1:02d}_{hid}.wav"

        if out_path.exists():
            print(f"  [{i+1}/19] Skip: {title}")
            ok += 1
            continue

        print(f"  [{i+1}/19] {title} ({len(content)} chars)")
        try:
            chunks = list(model.inference_instruct2(
                full_text,
                instruct_text=f"You are a helpful assistant. {INSTRUCTION}<|endofprompt|>",
                prompt_wav=PROMPT_WAV,
            ))
            if not chunks:
                print(f"    WARNING: No output")
                continue
            audio = chunks[0]["tts_speech"]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            torchaudio.save(str(out_path), audio.cpu(), SAMPLE_RATE)
            dur = audio.shape[-1] / SAMPLE_RATE
            print(f"    Saved {dur:.1f}s")
            ok += 1
        except Exception as e:
            print(f"    ERROR: {e}")
        torch.cuda.empty_cache()

    print(f"\nDone: {ok}/19")

if __name__ == "__main__":
    main()
