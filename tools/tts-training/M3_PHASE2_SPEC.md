# M3 Phase 2 Spec: data scaling 320 → ~1070 poems (task #10)

Holdout gap (m3_v4: train 21.6% vs holdout 30.3%) is now the main problem;
CosyVoice generation is nearly free, so scale the corpus. Source: the app's
own poem corpus `app/src/main/assets/poems_0..4.jsonl` (fields: `title`,
`author`, `dynasty`, `content` with `\n`).

## Script 1 (new): `scripts/m3_select_phase2.py` — runs LOCALLY

Runtime: `C:/Users/wsqsy/.conda/envs/open-webui/python.exe` (pypinyin
available). Paths below are relative to `tools/tts-training/` (resolve via
`Path(__file__).resolve().parents[1]` like other scripts).

Args (argparse, defaults):
- `--n` = 750 (target number of NEW poems)
- `--seed` = 42
- `--assets_dir` = `C:/Users/wsqsy/Documents/android/SuFei/app/src/main/assets`
- `--phase1` = `data/m3_phase1_poems.json` (existing 320, for dedup)
- `--out_gen` = `data/m3_phase2_poems.json` (JSON list of
  `{poem_id, text}` — same format as phase1, consumed by
  `m3_gen_cosyvoice.py`)
- `--out_skeleton` = `data/m3_phase2_skeleton.jsonl` (one JSON per line:
  `{poem_id, text, phoneme_ids}` — consumed by the prep script on rtx)

Steps:
1. Load all `poems_*.jsonl` from assets_dir.
2. Filter: `dynasty` in {唐代, 宋代}; flatten content (strip `\n`, spaces);
   flattened length 20-120 chars; every char of flattened content in
   CJK `[一-鿿]` or `{，。？！；：·}`; content ends with 。！？.
3. Build `text = f"{title}，{dynasty}·{author}。{content_flat}"` (same
   convention as phase1).
4. Dedup: parse phase1 texts (title = substring before the FIRST `，`,
   author = between `·` and the first `。`); skip candidates whose
   (normalized title, author) collides with phase1 or an earlier selection
   (normalize title by taking the part before ` / ` if present); also skip
   exact duplicate `content_flat`.
5. G2P → `phoneme_ids`: REUSE the pipeline in
   `scripts/generate_paddlespeech_distillation_data.py` (pypinyin → initials
   /finals with the y/w pseudo-initial and PYPINYIN_FINAL_FIX handling →
   `phones_to_ids` with the map from
   `models/paddlespeech_onnx/fastspeech2_csmsc_onnx_0.2.0/phone_id_map.txt`).
   Prefer importing the module; IF importing triggers side effects (check
   for module-level model loading), copy the needed functions verbatim
   instead and say so. DROP any candidate with unmapped phonemes (count
   and report).
6. Shuffle the filtered pool with `random.Random(seed)`, walk it and keep
   the first `n` candidates that pass G2P. Assign `poem_id = f"p2_{i:04d}"`
   in selection order.
7. Write both outputs. Print: pool size after filters, dedup-skips,
   G2P-drops, selected count, total flattened chars, estimated audio
   minutes at ~0.35 s/char.

## Script 2 (patch): `scripts/m3_prep_features.py`

Two backward-compatible extensions (defaults keep current behavior):

1. `--wav_dir` now accepts a COMMA-SEPARATED list of dirs; glob
   `*.wav` across all of them (candidates dict merges as before).
2. New `--extra_manifest` (default None): path (relative to ROOT) of a
   jsonl with rows `{poem_id, text, phoneme_ids}`. Rows are added to
   `manifest_recs` AND their poem_ids to `train_ids` (extra poems are all
   TRAIN; the fixed holdout 20 stays for comparability).

No other changes; the phase-1 320 poems get re-selected/re-extracted
identically when their wav dir is included (deterministic, acceptable).

## Acceptance

- py_compile both files with the open-webui python. Do NOT run anything.
- Only touch `scripts/m3_select_phase2.py` (new) and
  `scripts/m3_prep_features.py`.

## Execution plan (for reference, not part of implementation)

1. Run selection locally → ~750 new poems.
2. `m3_gen_cosyvoice.py m3_phase2_poems.json data/m3_cosyvoice_phase2 2`
   in WSL Ubuntu (~5-6 h, resumable).
3. scp phase2 wavs + skeleton to rtx; run prep with
   `--wav_dir <phase1>,<phase2> --extra_manifest data/m3_phase2_skeleton.jsonl`
   and `m3p2_*` output names.
4. m3_v5 = m3_v4 recipe on the merged manifest; eval via
   `eval_v10.py --m3 --stats data/m3p2_norm_stats.json` (same holdout 20).
