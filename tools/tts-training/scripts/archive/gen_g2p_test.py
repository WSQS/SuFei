"""Generate G2P test cases for Kotlin port validation."""
import sys, json
sys.path.insert(0, 'tools/tts-training/scripts')
from nar_stage1_e2e import text_to_phones, phones_to_ids, load_phone_id_map

phone_map = load_phone_id_map()

tests = [
    "春眠不觉晓",
    "床前明月光",
    "空山不见人",
    "千山鸟飞绝",
    "朝辞白帝彩云间",
]

results = []
for text in tests:
    phones = text_to_phones(text)
    ids = phones_to_ids(phones, phone_map)
    results.append({"text": text, "phones": phones, "ids": ids})
    print(f"{text}: {' '.join(phones)} ({len(ids)} ids)")

# Save as JSON for Kotlin test
with open("tools/tts-training/data/g2p_test_cases.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nSaved: g2p_test_cases.json")
