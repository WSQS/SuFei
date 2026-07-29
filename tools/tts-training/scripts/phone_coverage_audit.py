"""Phone coverage audit for unseen poems vs training set.

For each unseen poem, reports:
  - phone ID frequency in training set
  - unseen phone bigram hit rate
  - unseen phone trigram hit rate
  - zero-shot phone count
  - low-frequency phone count (train count <= 5)
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
sys.path.insert(0, str(ROOT / "scripts"))

from generate_paddlespeech_distillation_data import text_to_phonemes, load_phone_id_map

phone_map = load_phone_id_map()
id_to_phone = {v: k for k, v in phone_map.items()}

# Load training manifest
train_records = []
with open(ROOT / "data/paddle_distill_manifest.jsonl", encoding="utf-8") as f:
    for line in f:
        train_records.append(json.loads(line))

print(f"Training poems: {len(train_records)}")

# Build training phone statistics
train_phone_freq = Counter()
train_phone_bigrams = Counter()
train_phone_trigrams = Counter()

for rec in train_records:
    ids = rec["phoneme_ids"]
    for pid in ids:
        train_phone_freq[pid] += 1
    for i in range(len(ids) - 1):
        train_phone_bigrams[(ids[i], ids[i+1])] += 1
    for i in range(len(ids) - 2):
        train_phone_trigrams[(ids[i], ids[i+1], ids[i+2])] += 1

total_train_phones = sum(train_phone_freq.values())
unique_train_phones = len(train_phone_freq)
unique_train_bigrams = len(train_phone_bigrams)
unique_train_trigrams = len(train_phone_trigrams)

print(f"Training phone tokens: {total_train_phones}")
print(f"Unique phone IDs: {unique_train_phones}")
print(f"Unique phone bigrams: {unique_train_bigrams}")
print(f"Unique phone trigrams: {unique_train_trigrams}")

# Show least frequent phones
print(f"\nLeast frequent phone IDs in training (count < 10):")
for pid, cnt in sorted(train_phone_freq.items(), key=lambda x: x[1])[:20]:
    name = id_to_phone.get(pid, f"UNK_{pid}")
    print(f"  ID {pid:3d} ({name:6s}): {cnt:5d}")

# Load unseen manifest
unseen_records = []
with open(ROOT / "data/unseen_manifest.jsonl", encoding="utf-8") as f:
    for line in f:
        unseen_records.append(json.loads(line))

print(f"\n{'='*80}")
print(f"PER-POEM COVERAGE AUDIT")
print(f"{'='*80}")

for rec in unseen_records:
    pid = rec["poem_id"]
    ids = rec["phoneme_ids"]
    text = rec["text"]
    title = text.split("，")[0]

    # Phone frequency analysis
    phone_counts = Counter(ids)
    zero_shot = []  # phones not in training at all
    low_freq = []   # phones with <= 5 training occurrences
    for phone_id in phone_counts:
        train_cnt = train_phone_freq.get(phone_id, 0)
        name = id_to_phone.get(phone_id, f"UNK_{phone_id}")
        if train_cnt == 0:
            zero_shot.append((phone_id, name))
        elif train_cnt <= 5:
            low_freq.append((phone_id, name, train_cnt))

    # Bigram coverage
    unseen_bigrams = [(ids[i], ids[i+1]) for i in range(len(ids) - 1)]
    unseen_bg_hits = sum(1 for bg in unseen_bigrams if bg in train_phone_bigrams)
    unseen_bg_miss = len(unseen_bigrams) - unseen_bg_hits

    # Trigram coverage
    unseen_trigrams = [(ids[i], ids[i+1], ids[i+2]) for i in range(len(ids) - 2)]
    unseen_tg_hits = sum(1 for tg in unseen_trigrams if tg in train_phone_trigrams)
    unseen_tg_miss = len(unseen_trigrams) - unseen_tg_hits

    bg_hit_rate = unseen_bg_hits / max(len(unseen_bigrams), 1)
    tg_hit_rate = unseen_tg_hits / max(len(unseen_trigrams), 1)

    # Mel L1 from previous eval
    mel_l1_map = {
        "unseen_shanxing": 0.582,
        "unseen_shizhishang": 0.565,
        "unseen_shudaonan": 0.572,
        "unseen_chunjianghuayueye": 0.565,
        "unseen_pipaxing": 0.560,
        "unseen_jiangjinjiu": 0.571,
    }
    l1 = mel_l1_map.get(pid, 0)

    print(f"\n{'─'*60}")
    print(f"{pid} ({title})")
    print(f"  mel_len={rec['mel_len']}  n_phonemes={len(ids)}  mel_L1={l1:.3f}")
    print(f"  Zero-shot phones: {len(zero_shot)}")
    for ph_id, name in zero_shot:
        print(f"    ID {ph_id} ({name})")
    print(f"  Low-freq phones (train<=5): {len(low_freq)}")
    for ph_id, name, cnt in low_freq:
        print(f"    ID {ph_id} ({name}): train_count={cnt}")
    print(f"  Bigram coverage:  {unseen_bg_hits}/{len(unseen_bigrams)} = {bg_hit_rate:.1%} hit")
    print(f"  Trigram coverage: {unseen_tg_hits}/{len(unseen_trigrams)} = {tg_hit_rate:.1%} hit")

    # Show missed bigrams
    missed_bgs = [(bg, id_to_phone.get(bg[0], "?"), id_to_phone.get(bg[1], "?")) 
                  for bg in unseen_bigrams if bg not in train_phone_bigrams]
    if missed_bgs:
        print(f"  Missed bigrams ({len(missed_bgs)}):")
        for bg, p1, p2 in missed_bgs[:10]:
            print(f"    {p1} + {p2}")

# Summary table
print(f"\n{'='*80}")
print(f"SUMMARY TABLE")
print(f"{'='*80}")
print(f"{'Poem':<25} {'n_ph':>5} {'zero':>5} {'low_f':>6} {'bg_hit':>7} {'tg_hit':>7} {'L1':>6}")
print("─" * 65)
for rec in unseen_records:
    pid = rec["poem_id"]
    ids = rec["phoneme_ids"]
    title = rec["text"].split("，")[0]
    
    phone_counts = set(ids)
    zero_count = sum(1 for p in phone_counts if train_phone_freq.get(p, 0) == 0)
    low_count = sum(1 for p in phone_counts if 0 < train_phone_freq.get(p, 0) <= 5)
    
    unseen_bigrams = [(ids[i], ids[i+1]) for i in range(len(ids) - 1)]
    bg_hit = sum(1 for bg in unseen_bigrams if bg in train_phone_bigrams) / max(len(unseen_bigrams), 1)
    
    unseen_trigrams = [(ids[i], ids[i+1], ids[i+2]) for i in range(len(ids) - 2)]
    tg_hit = sum(1 for tg in unseen_trigrams if tg in train_phone_trigrams) / max(len(unseen_trigrams), 1)
    
    l1 = mel_l1_map.get(pid, 0)
    
    print(f"{title:<25} {len(ids):>5} {zero_count:>5} {low_count:>6} {bg_hit:>6.1%} {tg_hit:>6.1%} {l1:>.3f}")

# Also compute training set internal coverage for comparison
print(f"\n{'='*80}")
print(f"TRAINING SET INTERNAL BIGRAM/TRIGRAM COVERAGE")
print(f"{'='*80}")
# Leave-one-out: for each training poem, check if its bigrams are covered by others
loo_bg_rates = []
loo_tg_rates = []
for i, rec in enumerate(train_records):
    ids = rec["phoneme_ids"]
    others_bigrams = Counter()
    others_trigrams = Counter()
    for j, other in enumerate(train_records):
        if j == i:
            continue
        oids = other["phoneme_ids"]
        for k in range(len(oids) - 1):
            others_bigrams[(oids[k], oids[k+1])] += 1
        for k in range(len(oids) - 2):
            others_trigrams[(oids[k], oids[k+1], oids[k+2])] += 1
    
    poem_bgs = [(ids[k], ids[k+1]) for k in range(len(ids) - 1)]
    bg_hits = sum(1 for bg in poem_bgs if bg in others_bigrams)
    bg_rate = bg_hits / max(len(poem_bgs), 1)
    loo_bg_rates.append(bg_rate)
    
    poem_tgs = [(ids[k], ids[k+1], ids[k+2]) for k in range(len(ids) - 2)]
    tg_hits = sum(1 for tg in poem_tgs if tg in others_trigrams)
    tg_rate = tg_hits / max(len(poem_tgs), 1)
    loo_tg_rates.append(tg_rate)

import numpy as np
print(f"Leave-one-out bigram hit rate:  mean={np.mean(loo_bg_rates):.1%}  median={np.median(loo_bg_rates):.1%}  P10={np.percentile(loo_bg_rates, 10):.1%}")
print(f"Leave-one-out trigram hit rate: mean={np.mean(loo_tg_rates):.1%}  median={np.median(loo_tg_rates):.1%}  P10={np.percentile(loo_tg_rates, 10):.1%}")
print(f"(These show how well training poems cover each other)")
