"""Remove the "·"-as-"。" pause token from m3 manifests (m3_v6 data fix).

Root cause (poem_0027 doubled-王, persists in m3_v5): every training text is
"标题，朝代·作者。正文" and G2P maps · to the "。" phone. CosyVoice reads
朝代·作者 CONNECTED (no pause), so MAS must squeeze the "。" onto voiced
frames — the pause phone's acoustics pick up voiced leakage corpus-wide
(1× per poem), and at inference the predictor gives that "。" real-pause
duration (~17 frames), rendering a voiced pause + re-articulation
(唐代王王维). The faithful transcription has NO token at ·.

For each row: regenerate the (phone, char) sequence from `text` with the
ORIGINAL punct map; if it matches the stored phoneme_ids length, delete the
stored ids at positions where char == '·' (bit-safe for unaffected tokens);
otherwise fall back to full regeneration with the no-dot map. `durations`
(proportional placeholders) and `n_phonemes` are recomputed.

Runs locally (open-webui env, pypinyin). Usage:
  python m3_fix_dot_token.py --in data/X.jsonl --out data/Y.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_paddlespeech_distillation_data import (
    PUNCT_TO_PHONE,
    load_phone_id_map,
    phones_to_ids,
    text_to_phonemes,
)

NO_DOT_MAP = {k: v for k, v in PUNCT_TO_PHONE.items() if k != "·"}


def proportional_durations(n_phonemes, T):
    base = T // n_phonemes
    rem = T - base * n_phonemes
    durs = [base] * n_phonemes
    for i in range(rem):
        durs[i] += 1
    return durs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()

    phone_map = load_phone_id_map()
    n_rows = 0
    n_exact = 0      # regenerated ids identical to stored
    n_len_only = 0   # same length, some id values differ (pinyin drift)
    n_regen = 0      # length mismatch -> full no-dot regeneration
    removed_counts = {}

    with open(ROOT / args.in_path, encoding="utf-8") as fin, \
         open(ROOT / args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            stored = row["phoneme_ids"]
            phones = text_to_phonemes(row["text"])
            ids, _ = phones_to_ids(phones, phone_map)

            if len(ids) == len(stored):
                if ids == stored:
                    n_exact += 1
                else:
                    n_len_only += 1
                    print(f"  id-value drift (len ok): {row['poem_id']}")
                dot_pos = [i for i, (_, ch) in enumerate(phones)
                           if ch == "·"]
                new_ids = [v for i, v in enumerate(stored)
                           if i not in set(dot_pos)]
            else:
                n_regen += 1
                print(f"  LEN MISMATCH ({len(stored)}->{len(ids)}), "
                      f"full regen: {row['poem_id']}")
                phones_nd = text_to_phonemes(row["text"],
                                             punct_map=NO_DOT_MAP)
                new_ids, unmapped = phones_to_ids(phones_nd, phone_map)
                if unmapped:
                    print(f"    UNMAPPED {unmapped} in {row['poem_id']}")
                dot_pos = [1] * (len(stored) - len(new_ids))

            removed = len(stored) - len(new_ids)
            removed_counts[removed] = removed_counts.get(removed, 0) + 1

            row["phoneme_ids"] = new_ids
            if "n_phonemes" in row:
                row["n_phonemes"] = len(new_ids)
            if "durations" in row and "mel_len" in row:
                row["durations"] = proportional_durations(
                    len(new_ids), row["mel_len"])
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{args.in_path} -> {args.out_path}")
    print(f"  rows={n_rows} exact={n_exact} value-drift={n_len_only} "
          f"full-regen={n_regen}")
    print(f"  tokens-removed histogram: {sorted(removed_counts.items())}")


if __name__ == "__main__":
    main()
