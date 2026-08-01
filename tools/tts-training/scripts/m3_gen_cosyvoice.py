"""M3 Phase 0: CosyVoice3 poem generation smoke test.

Runs in WSL Ubuntu. Runtime: /home/sophomore/cosyvoice-venv/bin/python
CWD must be /home/sophomore/CosyVoice.

Generates two variants per poem (plain zero-shot + instruct recite),
resamples to 24 kHz mono, saves 16-bit PCM WAV.

Usage:
    /home/sophomore/cosyvoice-venv/bin/python scripts/m3_gen_cosyvoice.py \
        poems.json output_dir
"""
import os
import sys
import json
from pathlib import Path

import torchaudio
import torch

COSY_ROOT = os.environ.get('COSY_ROOT', '/home/sophomore/CosyVoice')
sys.path.insert(0, COSY_ROOT)
sys.path.append(COSY_ROOT + '/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

PROMPT_WAV = './asset/zero_shot_prompt.wav'
# CosyVoice3 unified prompt convention: '指令<|endofprompt|>参考音频转写'.
# The assert in the model requires the <|endofprompt|> marker to be present.
PROMPT_TEXT = '<|endofprompt|>希望你以后能够做的比我还好呦。'
PROMPT_TEXT_RECITE = ('用抑扬顿挫的语气深情朗诵这首古诗，语速缓慢，句读分明。'
                      '<|endofprompt|>希望你以后能够做的比我还好呦。')
TARGET_SR = 24000


def load_prompt(prompt_wav_path):
    """Load the prompt wav at 16 kHz — CosyVoice APIs expect prompt_speech_16k."""
    return load_wav(prompt_wav_path, 16000)


def collect_stream(gen):
    """Collect all yielded chunks' tts_speech and torch.cat along time."""
    chunks = []
    for j in gen:
        chunks.append(j['tts_speech'])
    return torch.cat(chunks, dim=1)


def save_wav(audio_tensor, out_path):
    """Resample from model sr to 24 kHz mono, save 16-bit PCM WAV."""
    audio_tensor = audio_tensor.squeeze(0) if audio_tensor.dim() == 2 else audio_tensor
    audio_tensor = audio_tensor.unsqueeze(0)  # (1, T)
    audio_tensor = (audio_tensor * 32767.0).clamp(-32768, 32767).to(torch.int16)
    torchaudio.save(str(out_path), audio_tensor, TARGET_SR)


def main():
    poems_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    n_candidates = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(poems_path, encoding='utf-8') as f:
        poems = json.load(f)

    print("Loading Fun-CosyVoice3-0.5B...")
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
    print("  loaded, sample_rate=%d" % cosyvoice.sample_rate)

    # This AutoModel build expects a file PATH for the prompt (it calls
    # load_wav internally; passing a tensor raises "Invalid file: tensor").
    prompt_wav_tensor = PROMPT_WAV
    instruct_fn = getattr(cosyvoice, 'inference_instruct2', None)
    if instruct_fn is not None:
        print("Instruct API: inference_instruct2")
    else:
        print("Instruct API: inference_instruct2 NOT found, will fall back")

    n_ok = 0
    n_fail = 0
    total_seconds = 0.0

    for poem in poems:
        pid = poem['poem_id']
        text = poem['text']

        # N candidates with the recite instruction (LLM sampling is
        # stochastic — each call yields a different take; best-of-N is
        # selected later by ASR CER on rtx).
        for k in range(n_candidates):
            out_path = output_dir / f"{pid}_c{k}.wav"
            if out_path.exists():
                n_ok += 1
                continue  # resumable
            try:
                gen = cosyvoice.inference_zero_shot(
                    text,
                    PROMPT_TEXT_RECITE,
                    prompt_wav_tensor,
                    stream=False,
                )
                audio = collect_stream(gen)
                if cosyvoice.sample_rate != TARGET_SR:
                    audio = torchaudio.functional.resample(
                        audio, cosyvoice.sample_rate, TARGET_SR
                    )
                save_wav(audio, out_path)
                dur = audio.shape[-1] / TARGET_SR
                total_seconds += dur
                n_ok += 1
                print("  %s: %.2fs" % (out_path.name, dur), flush=True)
            except Exception as e:
                n_fail += 1
                print("  %s FAILED: %s" % (out_path.name, e), flush=True)

    print("\nDone: n_ok=%d n_fail=%d total_audio=%.2fs" % (
        n_ok, n_fail, total_seconds))


if __name__ == '__main__':
    main()
