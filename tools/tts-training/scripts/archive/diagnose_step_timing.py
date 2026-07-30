"""Diagnose where time is spent in each training micro-batch with max_length=8192."""
import json
import os
import sys
import time
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MOSS-TTS-Nano"))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from finetuning.dataset import MossTTSNanoSFTDataset
from finetuning.common import load_jsonl_spec

MODEL_PATH = str(ROOT / "models" / "MOSS-TTS-Nano")
JSONL = str(ROOT / "data" / "train_with_codes.jsonl")
MAX_LENGTH = 8192
N_STEPS = 5


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] Setup...")

    records_paths, records = load_jsonl_spec(JSONL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"[{time.time()-t0:.1f}s] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation("sdpa")
    model.to("cuda")
    model.train()
    model.config.use_cache = False
    print(f"[{time.time()-t0:.1f}s] Model loaded, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    print(f"[{time.time()-t0:.1f}s] Building dataset (max_length={MAX_LENGTH})...")
    ds = MossTTSNanoSFTDataset(records, tokenizer=tokenizer, model_config=model.config, max_length=MAX_LENGTH)
    print(f"[{time.time()-t0:.1f}s] Dataset: {len(ds)} samples")

    # Find longest sample for worst case
    longest_idx = 0
    longest_len = 0
    for i in range(len(ds)):
        s = ds[i]
        sl = int(s["seq_len"])
        if sl > longest_len:
            longest_len = sl
            longest_idx = i
    print(f"[{time.time()-t0:.1f}s] Longest sample: idx={longest_idx}, seq_len={longest_len}")

    # Find a medium sample
    medium_idx = None
    for i in range(len(ds)):
        s = ds[i]
        sl = int(s["seq_len"])
        if 3000 <= sl <= 5000:
            medium_idx = i
            break
    print(f"[{time.time()-t0:.1f}s] Medium sample: idx={medium_idx}, seq_len={int(ds[medium_idx]['seq_len']) if medium_idx is not None else 'N/A'}")

    optimizer = AdamW(model.parameters(), lr=1e-5)

    from finetuning.sft import compute_supervised_loss, parse_channelwise_loss_weight
    channelwise_loss_weight = parse_channelwise_loss_weight("1,32", int(model.config.n_vq) + 1)

    # Test with collate_fn
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=ds.collate_fn, pin_memory=True)

    print(f"\n[{time.time()-t0:.1f}s] Starting {N_STEPS} micro-batch timing...")
    print(f"{'Step':>4} {'Idx':>4} {'SeqLen':>7} {'Data':>6} {'Fwd':>6} {'Loss':>6} {'Bwd':>6} {'Opt':>6} {'Total':>7} {'VRAM_GB':>8}")

    for step, batch in enumerate(dataloader):
        if step >= N_STEPS:
            break

        t_step = time.time()

        # Move data to GPU
        t_data = time.time()
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels = batch["labels"].to("cuda")
        seq_len = input_ids.shape[1]
        data_time = time.time() - t_data

        # Forward
        t_fwd = time.time()
        loss = compute_supervised_loss(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            channelwise_loss_weight=channelwise_loss_weight,
        )
        fwd_time = time.time() - t_fwd

        # Loss value
        t_loss = time.time()
        loss_val = loss.detach().item()
        loss_time = time.time() - t_loss

        # Backward
        t_bwd = time.time()
        loss.backward()
        bwd_time = time.time() - t_bwd

        # Optimizer (every 8 steps to simulate grad accumulation)
        opt_time = 0.0
        if (step + 1) % 8 == 0:
            t_opt = time.time()
            torch.cuda.synchronize()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_time = time.time() - t_opt

        total_time = time.time() - t_step
        vram = torch.cuda.memory_allocated() / 1e9

        print(f"{step:4d} {step:4d} {seq_len:7d} {data_time:6.2f} {fwd_time:6.2f} {loss_time:6.2f} {bwd_time:6.2f} {opt_time:6.2f} {total_time:7.2f} {vram:8.2f}")

        torch.cuda.synchronize()

    print(f"\n[{time.time()-t0:.1f}s] Done")


if __name__ == "__main__":
    main()
