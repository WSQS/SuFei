"""Forensic 3 & 4: Teacher-forced argmax + prefix-primed greedy."""
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = str(ROOT / "models" / "MOSS-TTS-Nano")
CODEC_PATH = str(ROOT / "models" / "MOSS-Audio-Tokenizer-Nano").replace("\\", "/")
CKPT = ROOT / "output" / "moss_poetry_sft_320_v4" / "checkpoint-epoch-3"
JSONL = ROOT / "data" / "train_with_codes.jsonl"

MOSS_AUDIO_TOKENIZER_TYPE = "moss-audio-tokenizer-nano"


def load_model(device, dtype):
    from safetensors.torch import load_file
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True)
    st = CKPT / "model.safetensors"
    if st.exists():
        state_dict = load_file(str(st))
        model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=dtype)
    if hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation("sdpa")
    model.eval()
    return model


def load_sample(jsonl_path, title_keyword):
    """Load a training sample matching title keyword."""
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            if title_keyword in r["text"]:
                return r
    return None


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load model + tokenizer + dataset
    from transformers import AutoTokenizer
    sys_path = str(ROOT / "MOSS-TTS-Nano")
    import sys
    sys.path.insert(0, sys_path)
    from finetuning.dataset import MossTTSNanoSFTDataset
    from finetuning.sft import compute_supervised_loss, parse_channelwise_loss_weight

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = load_model(device, dtype)

    # Find 静夜思 in training data
    record = load_sample(JSONL, "静夜思")
    if not record:
        print("ERROR: 静夜思 not found in training data")
        return

    # Resolve audio path
    audio_path = record["audio"]
    if not Path(audio_path).is_absolute():
        audio_path = str((ROOT / "data" / audio_path).resolve())
    record["audio"] = audio_path

    ds = MossTTSNanoSFTDataset([record], tokenizer=tokenizer, model_config=model.config, max_length=8192)
    sample = ds[0]
    input_ids = sample["full_input_ids"].unsqueeze(0).to(device)
    prompt_len = int(sample["prompt_length"])
    seq_len = int(sample["seq_len"])

    print(f"Sample: 静夜思")
    print(f"  prompt_len={prompt_len}, seq_len={seq_len}")
    print(f"  audio_frames={seq_len - prompt_len - 1}")  # -1 for EOS

    n_audio_frames = seq_len - prompt_len - 1
    cfg = model.config
    n_vq = cfg.n_vq

    # === Forensic 3: Teacher-forced argmax ===
    # Feed the full sequence, get logits at each audio frame position,
    # take argmax, and decode the argmax tokens
    print(f"\n{'='*60}")
    print(f"Forensic 3: Teacher-forced argmax decode")
    print(f"{'='*60}")

    model.eval()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=dtype):
            outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)

    global_hidden = outputs.global_hidden_states  # (1, seq_len, hidden)
    print(f"  global_hidden shape: {global_hidden.shape}")

    # The local transformer processes each position and predicts n_vq+1 channels
    # We need to run the local transformer to get actual token predictions
    # Use the model's own generation internals for this

    # Actually, let's use a simpler approach: compare the ground truth audio tokens
    # with what the model would predict via argmax at each step
    # The model's generate function with teacher forcing isn't directly exposed,
    # but we can check: if we feed the full sequence including real audio tokens,
    # what does the model predict at each audio position?

    # Get the full input and look at the audio token predictions
    # The local_transformer makes the final predictions
    base_model = model
    batch_size, sl, hs = global_hidden.shape

    # Build local inputs exactly like compute_supervised_loss does
    flat_hidden = global_hidden.reshape(batch_size * sl, hs)
    local_dtype = base_model.local_transformer.ln_f.weight.dtype
    flat_hidden = flat_hidden.to(dtype=local_dtype)

    # Get ground truth labels from input
    full_ids = sample["full_input_ids"]  # (seq_len, n_vq+1)
    # Shift by 1 for next-token prediction
    labels = full_ids[1:].clone()  # (seq_len-1, n_vq+1)

    # Run local transformer on the full sequence
    local_inputs = torch.zeros((batch_size * sl, n_vq + 1, hs), dtype=local_dtype, device=device)
    local_inputs[:, 0, :] = flat_hidden

    # Text channel (channel 0)
    text_targets = labels[:, 0] if labels.shape[0] == sl - 1 else full_ids[1:, 0]
    # Pad to match sl
    safe_text = text_targets[:sl].masked_fill(text_targets[:sl].lt(0), int(cfg.pad_token_id))
    local_inputs[:sl-1, 0, :] = base_model.transformer.wte(safe_text.to(device))

    # Audio channels (1..n_vq)
    for ch in range(n_vq):
        ch_targets = full_ids[1:, ch + 1] if ch + 1 < n_vq + 1 else torch.zeros(sl - 1, dtype=torch.long)
        ch_targets = ch_targets[:sl-1].to(device)
        valid = (ch_targets >= 0) & (ch_targets < base_model.audio_embeddings[ch].num_embeddings)
        safe = ch_targets.masked_fill(~valid, 0)
        emb = base_model.audio_embeddings[ch](safe)
        local_inputs[:sl-1, ch + 1, :] = emb.to(dtype=local_dtype)

    # Forward through local transformer
    local_result = base_model.local_transformer(inputs_embeds=local_inputs)
    local_hidden = local_result.last_hidden_state if hasattr(local_result, 'last_hidden_state') else local_result[0]
    print(f"  local_hidden shape: {local_hidden.shape}")
    # local_out shape: (sl, n_vq+1, hidden) or similar

    # Get logits from heads
    audio_frame_preds = []
    for frame_idx in range(prompt_len, prompt_len + min(n_audio_frames, 50)):  # First 50 audio frames
        frame_preds = []
        for ch in range(n_vq):
            logits = base_model.audio_lm_heads[ch](local_hidden[frame_idx, ch + 1])
            pred = logits.argmax().item()
            gt = full_ids[frame_idx, ch + 1].item()
            top3 = torch.topk(logits, 3).indices.tolist()
            frame_preds.append((gt, pred, top3))
        audio_frame_preds.append(frame_preds)

    print(f"\n  First 10 audio frames (GT vs Argmax, CB0-3):")
    for i, fp in enumerate(audio_frame_preds[:10]):
        cbs = []
        for ch, (gt, pred, top3) in enumerate(fp[:4]):
            match = "OK" if gt == pred else "XX"
            cbs.append(f"CB{ch}: gt={gt} pred={pred} {match}")
        print(f"    Frame {i}: {' | '.join(cbs)}")

    # Accuracy
    correct = sum(1 for fp in audio_frame_preds for gt, pred, _ in fp if gt == pred)
    total = sum(1 for fp in audio_frame_preds for _, _, _ in fp)
    print(f"\n  Teacher-forced argmax accuracy (first 50 frames, all codebooks): {correct}/{total} = {correct/total*100:.1f}%")

    # Check if greedy-absorbing tokens appear
    absorbing_tokens = {0: 482, 1: 194, 2: 670, 3: 241}
    print(f"\n  Are absorbing tokens (482/194/670/241) ever the argmax?")
    for ch in range(4):
        absorb = absorbing_tokens[ch]
        count = sum(1 for fp in audio_frame_preds for gt, pred, top3 in [fp[ch]] if pred == absorb)
        gt_count = sum(1 for fp in audio_frame_preds for gt, pred, top3 in [fp[ch]] if gt == absorb)
        print(f"    CB{ch}: argmax={absorb} in {count}/50 frames, ground truth has it in {gt_count}/50")

    # === Forensic 4: Prefix-primed generation ===
    print(f"\n{'='*60}")
    print(f"Forensic 4: Prefix-primed greedy generation")
    print(f"{'='*60}")

    # We can't easily inject real audio prefix into model.inference(),
    # but we can check: does giving the model a short prompt with prompt_text help?
    # Actually, the continuation mode already gives full text. The question is
    # whether giving real audio codes as prefix helps.

    # Instead, let's check the EOS probability at each frame
    # by looking at the logits for audio_end_token_id
    eos_id = cfg.audio_end_token_id
    print(f"\n  EOS token id: {eos_id}")
    print(f"  Checking text channel logits for EOS at audio frame positions:")

    # The text head predicts text tokens including EOS
    text_logits = base_model.text_lm_head(local_hidden[:, 0, :])  # (sl, vocab)
    eos_probs = torch.softmax(text_logits.float(), dim=-1)

    print(f"\n  Frame | EOS_prob | EOS_rank | Top1_token")
    for frame_idx in range(prompt_len, min(prompt_len + 20, sl)):
        eos_prob = eos_probs[frame_idx, eos_id].item()
        eos_rank = (eos_probs[frame_idx] > eos_probs[frame_idx, eos_id]).sum().item()
        top1 = text_logits[frame_idx].argmax().item()
        gt = full_ids[frame_idx, 0].item()
        print(f"    {frame_idx-prompt_len:3d} | {eos_prob:.4f} | {eos_rank+1:5d} | top1={top1} (gt={gt})")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
