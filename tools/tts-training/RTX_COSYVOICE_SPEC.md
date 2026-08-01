# RTX CosyVoice Environment Spec (native Windows conda — no WSL on rtx)

## Goal

Set up Fun-CosyVoice3-0.5B inference on the rtx machine (Windows 11, RTX
4090, conda at `C:\Users\wehao\anaconda3`) so poem generation can run there
instead of (or in parallel with) the local WSL Ubuntu box. rtx has NO WSL,
so this is the native-Windows route: pynini comes from conda-forge (win-64
builds exist; this is the official CosyVoice README guidance for Windows),
WeTextProcessing installs with `--no-deps` on top.

The repo + model weights (~10 GB) are being scp'd to `E:\CosyVoice`
separately (LAN copy of the WORKING local install — includes
`pretrained_models/Fun-CosyVoice3-0.5B` and `third_party/Matcha-TTS`).

The working local venv's exact package set is in
`rtx_setup_cosy_freeze.txt` (pip freeze, Linux). Torch is 2.5.1+cu124;
rtx has CUDA toolkit 12.4 / driver 12.9 — same wheels family.

## Deliverables (LOCAL FILES ONLY — do not run anything remote)

Create a directory `rtx_setup/` under `tools/tts-training/` with:

### 1. `rtx_setup/rtx_cosyvoice_requirements.txt`

Derived from `rtx_setup_cosy_freeze.txt` by EXCLUDING:
- `torch==`, `torchaudio==`, `torchvision==` (installed separately from
  the cu124 index; torchvision is not needed for inference at all)
- `nvidia-*` (Linux-only CUDA runtime wheels; Windows torch bundles CUDA)
- `triton*`, `deepspeed*`, `flash*` (Linux-only; training-only)
- `pynini*` (comes from conda-forge)
- `WeTextProcessing*` (installed separately with --no-deps)
- any line containing `@` (direct/file references, not portable)
- any of these if present (Linux-only or irrelevant): `uvloop`, `xformers`,
  `bitsandbytes`, `ray`, `ttsfrd*`, `tensorrt*`
Keep everything else with exact pins (they coexisted in one working env).
Add a header comment listing what was excluded and why (one line each).

### 2. `rtx_setup/setup_cosyvoice.bat`

A self-contained batch script that will be scp'd to `E:\` on rtx and run
via ssh under cmd. Requirements:

- `@echo off`, every step echoes a `[STEP n] ...` marker line before
  running, and the script ends with `SETUP_DONE` on success or
  `SETUP_FAILED step=<n> code=<errorlevel>` on the first failure (use
  `|| goto :err_n` per step or a CALL-based error pattern — each failure
  must identify WHICH step died).
- Steps:
  1. `call C:\Users\wehao\anaconda3\condabin\conda.bat create -n cosyvoice python=3.10 -y`
     (idempotent-ish: if the env already exists this errors — precede with
     `conda.bat env remove -n cosyvoice -y` guarded to not fail when absent:
     append `& ver>nul` trick or `|| ver>nul` so a missing env doesn't abort).
  2. `call ...conda.bat install -n cosyvoice -c conda-forge pynini=2.1.5 -y`
  3. `set PY=C:\Users\wehao\anaconda3\envs\cosyvoice\python.exe`
  4. Torch (Aliyun mirror first, official fallback):
     `%PY% -m pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 -f https://mirrors.aliyun.com/pytorch-wheels/cu124/`
     `|| %PY% -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124`
  5. `%PY% -m pip install -r E:\rtx_cosyvoice_requirements.txt -i https://mirrors.aliyun.com/pypi/simple/`
  6. `%PY% -m pip install WeTextProcessing==1.0.3 --no-deps -i https://mirrors.aliyun.com/pypi/simple/`
  7. Verify:
     `set KMP_DUPLICATE_LIB_OK=TRUE`
     `%PY% -c "import torch; print('TORCH_OK', torch.__version__, torch.cuda.is_available())"`
     `cd /d E:\CosyVoice`
     `set COSY_ROOT=E:\CosyVoice`
     `%PY% -c "import sys, os; sys.path.insert(0, os.environ['COSY_ROOT']); sys.path.append(os.path.join(os.environ['COSY_ROOT'], 'third_party', 'Matcha-TTS')); from cosyvoice.cli.cosyvoice import AutoModel; print('IMPORT_OK')"`
     (import only — do NOT load the model in the batch; the real smoke test
     runs separately).

### 3. Patch `scripts/m3_gen_cosyvoice.py` (backward compatible)

Replace the hardcoded `COSY_ROOT = '/home/sophomore/CosyVoice'` with:

```python
COSY_ROOT = os.environ.get('COSY_ROOT', '/home/sophomore/CosyVoice')
```

(add `import os`; `PROMPT_WAV = './asset/zero_shot_prompt.wav'` stays
relative — the runner cd's into the repo root on either machine). No other
changes. Default preserves current WSL behavior exactly.

## Acceptance

- py_compile passes for `scripts/m3_gen_cosyvoice.py` (open-webui python).
- The .bat is plain ASCII, CRLF or LF both fine, no PowerShell.
- Do NOT execute anything on rtx and do NOT run the .bat — it is executed
  separately via fdx. Only create/modify the three files listed.
