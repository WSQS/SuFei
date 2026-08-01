"""M3 Phase 2: select ~750 NEW poems for data scaling (task #10).

Runs LOCALLY with the open-webui conda env (pypinyin available).
Selects poems from the app's assets corpus, deduplicates against phase-1,
runs G2P via the existing PaddleSpeech pipeline, and writes two outputs:
  - m3_phase2_poems.json    (consumed by m3_gen_cosyvoice.py)
  - m3_phase2_skeleton.jsonl (consumed by m3_prep_features.py --extra_manifest)
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# G2P reuse: generate_paddlespeech_distillation_data has NO module-level side
# effects (only Path constants and `import numpy`; all model loading happens
# inside main()). Safe to import the two pure functions directly.
from generate_paddlespeech_distillation_data import (
    load_phone_id_map,
    phones_to_ids,
    text_to_phonemes,
)

FS2_DIR = ROOT / "models" / "paddlespeech_onnx" / "fastspeech2_csmsc_onnx_0.2.0"

CJK_RE = re.compile(r"^[\u4e00-\u9fff，。？！；：·]+$")
ALLOWED_PUNCT_END = "。！？"


def flatten_content(content):
    return content.replace("\n", "").replace(" ", "")


def passes_filter(rec):
    if rec.get("dynasty") not in ("唐代", "宋代"):
        return False
    flat = flatten_content(rec["content"])
    if not (20 <= len(flat) <= 120):
        return False
    if not CJK_RE.match(flat):
        return False
    if flat[-1] not in ALLOWED_PUNCT_END:
        return False
    return True


def build_text(title, dynasty, author, content_flat):
    return f"{title}，{dynasty}·{author}。{content_flat}"


def normalize_title(title):
    return title.split(" / ")[0].strip()


def parse_phase1_for_dedup(phase1_texts):
    seen = set()
    for text in phase1_texts:
        title = text.split("，", 1)[0]
        author = ""
        if "·" in text and "。" in text:
            between = text.split("·", 1)[1]
            author = between.split("。", 1)[0]
        seen.add((normalize_title(title), author))
    return seen


def main():
    parser = argparse.ArgumentParser(
        description="M3 Phase 2: select NEW poems for data scaling")
    parser.add_argument("--n", type=int, default=750,
                        help="Target number of NEW poems")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--assets_dir",
        default="C:/Users/wsqsy/Documents/android/SuFei/app/src/main/assets")
    parser.add_argument("--phase1", default="data/m3_phase1_poems.json")
    parser.add_argument("--out_gen", default="data/m3_phase2_poems.json")
    parser.add_argument("--out_skeleton",
                        default="data/m3_phase2_skeleton.jsonl")
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir)

    # ── Step 1: Load all poems_*.jsonl ──
    candidates = []
    for jl in sorted(assets_dir.glob("poems_*.jsonl")):
        with open(jl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                candidates.append(json.loads(line))

    # ── Step 2: Filter ──
    filtered = []
    for rec in candidates:
        if passes_filter(rec):
            flat = flatten_content(rec["content"])
            filtered.append({
                "title": rec["title"],
                "author": rec["author"],
                "dynasty": rec["dynasty"],
                "content_flat": flat,
            })
    pool_size = len(filtered)

    # ── Step 4: Dedup against phase-1 ──
    phase1_path = ROOT / args.phase1
    phase1_texts = []
    if phase1_path.exists():
        with open(phase1_path, encoding="utf-8") as f:
            phase1_data = json.load(f)
        phase1_texts = [item["text"] for item in phase1_data]

    seen = parse_phase1_for_dedup(phase1_texts)
    seen_contents = {t.split("。", 1)[-1] for t in phase1_texts}
    dedup_skips = 0
    deduped = []
    for cand in filtered:
        key = (normalize_title(cand["title"]), cand["author"])
        if key in seen or cand["content_flat"] in seen_contents:
            dedup_skips += 1
            continue
        seen.add(key)
        seen_contents.add(cand["content_flat"])
        deduped.append(cand)

    # ── Step 5: G2P setup ──
    phone_map = load_phone_id_map()

    # ── Step 6: Shuffle, walk, keep first n that pass G2P ──
    rng = random.Random(args.seed)
    rng.shuffle(deduped)

    selected = []
    g2p_drops = 0
    total_chars = 0
    for cand in deduped:
        if len(selected) >= args.n:
            break
        text = build_text(cand["title"], cand["dynasty"],
                          cand["author"], cand["content_flat"])
        phonemes = text_to_phonemes(text)
        ids, unmapped = phones_to_ids(phonemes, phone_map)
        if unmapped:
            g2p_drops += 1
            continue
        idx = len(selected)
        poem_id = f"p2_{idx:04d}"
        selected.append({"poem_id": poem_id, "text": text,
                         "phoneme_ids": ids})
        total_chars += len(cand["content_flat"])

    # ── Step 7: Write outputs ──
    out_gen_path = ROOT / args.out_gen
    out_gen_path.parent.mkdir(parents=True, exist_ok=True)
    gen_data = [{"poem_id": s["poem_id"], "text": s["text"]}
                for s in selected]
    with open(out_gen_path, "w", encoding="utf-8") as f:
        json.dump(gen_data, f, ensure_ascii=False, indent=2)

    out_skel_path = ROOT / args.out_skeleton
    with open(out_skel_path, "w", encoding="utf-8") as f:
        for s in selected:
            row = {"poem_id": s["poem_id"], "text": s["text"],
                   "phoneme_ids": s["phoneme_ids"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    est_minutes = total_chars * 0.35 / 60.0
    print(f"Pool after filters:   {pool_size}")
    print(f"Dedup skips:          {dedup_skips}")
    print(f"G2P drops:            {g2p_drops}")
    print(f"Selected:             {len(selected)}")
    print(f"Total flattened chars: {total_chars}")
    print(f"Est. audio minutes:   {est_minutes:.1f}")
    print(f"Outputs:")
    print(f"  {out_gen_path}")
    print(f"  {out_skel_path}")


if __name__ == "__main__":
    main()
