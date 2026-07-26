"""Dump ASR transcriptions vs reference text for manual inspection."""
import json, re, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "MOSS-TTS-Nano")

import sys
sys.path.insert(0, sys_path)

from funasr import AutoModel
from opencc import OpenCC

PUNCT = re.compile(r"[，。！？；：、,.!?;:\"'""''（）()\[\]【】《》〈〉…—\-．\.\n\r\s\u3000]")
cc = OpenCC("t2s")

def normalize(text):
    text = cc.convert(text)
    return PUNCT.sub("", text)

with open(ROOT / "data" / "test_set_v1.json", encoding="utf-8") as f:
    test_set = json.load(f)

audio_dir = ROOT / "data" / "audio"

print("Loading ASR...")
model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    device="cuda",
    disable_update=True,
)

print("\nPoem                 | Ref (content)                          | ASR raw                                | ASR norm                              | Match?")
print("-" * 200)

for s in test_set["samples"]:
    wav_name = s.get("teacher_wav")
    if not wav_name:
        continue
    wav_path = audio_dir / wav_name
    if not wav_path.exists():
        continue

    result = model.generate(input=str(wav_path), batch_size_s=300)
    asr_raw = result[0].get("text", "") if result else ""
    asr_norm = normalize(asr_raw)

    ref_content = s["content"]
    ref_norm = normalize(ref_content)

    # Check if ref is a substring of asr_norm (title prefix would cause this)
    ref_in_asr = ref_norm in asr_norm if ref_norm else False
    asr_in_ref = asr_norm in ref_norm if asr_norm else False

    print(f"{s['title']:20s} | {ref_norm[:35]:38s} | {asr_norm[:38]:40s} | ref_in_asr={ref_in_asr} asr_len={len(asr_norm)} ref_len={len(ref_norm)}")
