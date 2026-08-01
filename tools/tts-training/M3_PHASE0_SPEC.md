# M3 Phase 0 Spec: CosyVoice3 poem generation smoke + ASR gate + vocoder ceiling

## Goal

Answer three unknowns before scaling M3 (new-data-source training via the v10f
MAS recipe): (1) can Fun-CosyVoice3-0.5B recite classical poems correctly and
with better prosody than the PaddleSpeech teacher; (2) what fraction passes an
ASR gate; (3) what is the copy-synthesis ceiling of CosyVoice audio through
the EXISTING feature-extraction + HiFiGAN CSMSC path (decides whether M3
training can reuse the current target space or needs vocoder adaptation
first).

## Script 1: `scripts/m3_gen_cosyvoice.py` (runs in WSL Ubuntu)

Runtime: `/home/sophomore/cosyvoice-venv/bin/python`, CWD must be
`/home/sophomore/CosyVoice`, and the script must do
`sys.path.append('third_party/Matcha-TTS')` before cosyvoice imports.
CUDA is available (torch 2.5.1+cu124).

- Args: `argv[1]` = poems JSON (list of `{poem_id, text}`),
  `argv[2]` = output dir (create if missing).
- Load `AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')`
  (from `cosyvoice.cli.cosyvoice import AutoModel`).
- Per poem, generate TWO variants:
  - `plain`: `inference_zero_shot(text, prompt_text, prompt_wav)` with the
    stock prompt `./asset/zero_shot_prompt.wav`, prompt text
    `希望你以后能够做的比我还好呦。`
  - `recite`: instruct-style with instruction
    `用抑扬顿挫的语气深情朗诵这首古诗，语速缓慢，句读分明。`
    API discovery at runtime: prefer `inference_instruct2(text, instruct,
    prompt_wav)` if the method exists; otherwise fall back to another
    available instruct method or zero-shot (PRINT which API was used).
- `stream=False`; collect ALL yielded chunks' `j['tts_speech']` and
  `torch.cat` along time.
- Resample from `cosyvoice.sample_rate` to 24000 Hz mono, save 16-bit PCM WAV
  as `{poem_id}_{variant}.wav` in the output dir.
- Per-poem try/except: log failures and continue (AR models can fail on some
  inputs). Print per-file duration in seconds and a final summary
  (n_ok / n_fail, total audio seconds).

## Script 2: `scripts/m3_phase0_assess.py` (runs on rtx)

Follows the conventions of `eval_v10.py` (ROOT = `E:/sufei-training`, imports
`load_asr_model`/`transcribe` from `eval_tts` and `cer_detail` from
`eval_unified`).

- Args: `--wav_dir` (default `E:/sufei-training/data/m3_cosyvoice_phase0`),
  `--poems` (default `E:/sufei-training/data/m3_phase0_poems.json`).
- For each `{poem_id}_{variant}.wav`:
  1. **direct CER**: transcribe the wav, `cer_detail(text, asr_text)`.
  2. **ceiling CER**: extract log-mel from the wav using THE SAME mel
     extraction function/params used to build `paddle_distill_features_v3`
     (FIND it among the repo scripts — e.g. the v3 rebuild scripts
     `gen_new_129_distill.py` / `build_train_300.py` / related; REUSE the
     exact code, do not reinvent; print which source you used) → run
     HiFiGAN ONNX
     (`E:/sufei-training/models/paddlespeech_onnx/hifigan_csmsc_onnx_0.2.0/hifigan_csmsc.onnx`,
     input name `logmel`, per `eval_v10.py`) → save vocoded wav as
     `{poem_id}_{variant}_voc.wav` → transcribe → CER.
- Print a table: `poem_id | variant | dur_s | direct_CER | ceiling_CER`,
  then per-variant means for both CER columns, plus n(100% CER).

## Acceptance

- `python -m py_compile` passes for both scripts (use the Windows python
  `C:/Users/wsqsy/.conda/envs/open-webui/python.exe` for the syntax check of
  both; Script 1 will only RUN in WSL — do not run either script).
- Script 1 must not import anything beyond: sys/json/pathlib, torch,
  torchaudio, cosyvoice.
- Only create `scripts/m3_gen_cosyvoice.py` and `scripts/m3_phase0_assess.py`.
  Do not modify other files. Do not run generation or assessment.

---

# M3 Phase 1 Addendum: data prep + cold-start training (user-approved)

Phase 0 verdict: source switch approved; recite wording kept for now.
Generation of 320 poems x 2 candidates is running separately. Two pieces
remain.

## 1. New script `scripts/m3_prep_features.py` (runs on rtx)

Follows eval_v10.py conventions (ROOT = `E:/sufei-training`, ASR via
`eval_tts.load_asr_model/transcribe`, `eval_unified.cer_detail`).

Args (argparse, all with defaults):
- `--wav_dir` = `E:/sufei-training/data/m3_cosyvoice_phase1`
  (files `{poem_id}_c{k}.wav`, 24 kHz mono)
- `--train_manifest` = `data/train_300_manifest_v3.jsonl`,
  `--holdout_manifest` = `data/holdout_20_manifest_v3.jsonl`
  (source of poem_id / text / phoneme_ids — reuse them verbatim)
- `--out_feat_dir` = `data/m3_features`,
  `--out_train` = `data/m3_train_manifest.jsonl`,
  `--out_holdout` = `data/m3_holdout_manifest.jsonl`,
  `--out_stats` = `data/m3_norm_stats.json`
- `--max_cer` = 0.45 (quality gate)

Steps:
1. **Best-of-N selection**: ASR-transcribe every candidate wav; per poem pick
   the lowest CER (tie -> lowest k). Print a selection table
   (poem_id, chosen k, CER per candidate). Drop poems whose best CER >
   `--max_cer`; print the dropped list and counts.
2. **Feature extraction** for each selected wav:
   - **mel**: the CALIBRATED convention — magnitude STFT (power=1) with
     n_fft=2048, hop=300, win=1200, hann; mel basis sr=24000, n_mels=80,
     fmin=80, fmax=7600; `np.log10(np.maximum(mel, 1e-10))`, shape (T, 80)
     float32. (Same as the patched `extract_mel` in `m3_phase0_assess.py` —
     import it from there or copy verbatim.)
   - **f0**: pyworld dio+stonemask with EXPLICIT
     `frame_period=300/24000*1000 = 12.5` (ms) so F0 is natively at the mel
     frame rate — NO index-based resampling (the historical //256 bug class).
     Pad/trim to exactly T frames (unvoiced=0).
   - **energy**: per-frame log10 of linear-domain mel sum:
     `np.log10(np.maximum((10.0 ** mel).sum(axis=1), 1e-10))` — monotone,
     not floor-dominated. (New dataset => new stats, so the definition change
     vs v3 is safe.)
   - Save `{poem_id}.npz` with keys `mel`, `f0`, `energy` (float32).
3. **Manifest rows**: copy `poem_id`, `text`, `phoneme_ids` from the v3
   manifest row; set `mel_len` = T; set `durations` = proportional integers
   (T split as evenly as possible over the phonemes, summing EXACTLY to T) —
   placeholder for TTSDataset compatibility only (cold-start never expands
   with them; dur_corr diagnostics vs them are meaningless — that is
   accepted).
4. **Norm stats** (train split only): per-bin mel mean/std; f0 mean/std of
   log(f0[f0>0]); energy mean/std. Same JSON schema as
   `train_300_norm_stats_v3.json`.

## 2. `scripts/nar_train_align.py`: add `--cold_start` flag

For data with no usable teacher durations:

- While the phase gate is CLOSED (`use_mas` False):
  `total_loss = W_ALIGN*align_loss + w_fsum*fsum + W_DUR*dur_loss` ONLY.
  Skip length-regulate/decoder/mel/pitch/energy entirely (do not compute
  them — saves time and avoids garbage expansion). Log mel/pitch/energy
  as 0.0 in this phase.
- While OPEN: identical to current v10d/f behavior with
  `dur_for_expand = mas_durations`.
- `durations_gt` must NEVER be used for expansion when `--cold_start`.
- Non-cold-start behavior must remain bit-identical to today.

## Acceptance (Phase 1)

- py_compile passes for `m3_prep_features.py` and `nar_train_align.py`
  (Windows python as above). Do not run anything.
- Only touch `scripts/m3_prep_features.py` (new) and
  `scripts/nar_train_align.py`.
