"""Check: with garbage durations [560, 2, 2, ...], what does the model
actually see during training? Can it still learn?

Key question: GT durations [560, 2, 2, ...] sum to mel_len (687), so
the length regulator produces 687 frames. But frames 0-559 all get
the SAME phoneme[0] embedding. The mel target at those frames is
actual speech content. How can the model learn?

Hypothesis: with gt_variance, pitch/energy embeddings carry per-frame
prosody information that compensates for the wrong phoneme expansion.
The model learns pitch/energy → mel, not phoneme → mel.
"""
import sys, json, math, numpy as np, torch
sys.path.insert(0, r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training\scripts")
from nar_fastspeech2 import FastSpeech2
from pathlib import Path

ROOT = Path(r"C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training")
STATS = json.load(open(ROOT / "data" / "train_300_norm_stats.json"))
ckpt = torch.load(str(ROOT / "checkpoints" / "D300e2e_step24000_slim.pt"),
                   map_location="cpu", weights_only=False)
model = FastSpeech2(vocab_size=268, dropout=0.0)
model.load_state_dict(ckpt["model"])
model.eval()

rec = [json.loads(l) for l in open(ROOT / "data" / "train_300_manifest.jsonl", encoding="utf-8")][0]
pid = rec["poem_id"]
npz = np.load(str(ROOT / "data" / "paddle_distill_features" / f"{pid}.npz"))

dur_gt = torch.tensor([rec["durations"]], dtype=torch.long)
phone_ids = torch.tensor([rec["phoneme_ids"]], dtype=torch.long)
mel_gt = torch.tensor(npz["mel"].astype(np.float32)).unsqueeze(0)
f0_gt = torch.tensor(npz["f0"].astype(np.float32)).unsqueeze(0)
e_gt = torch.tensor(npz["energy"].astype(np.float32)).unsqueeze(0)

mel_mean = torch.tensor(STATS["mel_mean"])
mel_std = torch.tensor(STATS["mel_std"])
f0_norm = torch.where(f0_gt > 0,
    (torch.log(f0_gt.clamp(min=1)) - STATS["f0_mean"]) / STATS["f0_std"],
    torch.zeros_like(f0_gt))
e_norm = (e_gt - STATS["energy_mean"]) / STATS["energy_std"]

# Show what the length regulator produces
with torch.no_grad():
    x = model.embedding(phone_ids) * math.sqrt(model.d_model)
    x = model.pos_enc(x)
    for layer in model.encoder_layers:
        x = layer(x)
    
    expanded = model.length_regulator(x, dur_gt)  # [1, 687, 256]
    
    # Check: are frames 0-559 actually all the same?
    frame_diffs = (expanded[0, 1:100] - expanded[0, 0:99]).norm(dim=1)
    print(f"Frame-to-frame L2 distance (frames 0-99):")
    print(f"  mean={frame_diffs.mean():.4f} min={frame_diffs.min():.4f} max={frame_diffs.max():.4f}")
    print(f"  frames 0-1 diff: {frame_diffs[0]:.6f}")
    print(f"  frames 559-560 diff: {(expanded[0, 560] - expanded[0, 559]).norm():.6f}")
    print(f"  frames 560-561 diff: {(expanded[0, 561] - expanded[0, 560]).norm():.6f}")
    
    # What info does the decoder get per frame?
    pitch_emb = model.pitch_embed(f0_norm[:, :expanded.size(1)].unsqueeze(-1))
    energy_emb = model.energy_embed(e_norm[:, :expanded.size(1)].unsqueeze(-1))
    
    total_input = expanded + pitch_emb + energy_emb
    
    # Compare contributions
    phoneme_norm = expanded[0].norm(dim=1).mean().item()
    pitch_norm = pitch_emb[0].norm(dim=1).mean().item()
    energy_norm = energy_emb[0].norm(dim=1).mean().item()
    
    print(f"\nDecoder input contribution (L2 norm per frame):")
    print(f"  Phoneme (expanded): {phoneme_norm:.2f}")
    print(f"  Pitch embed:        {pitch_norm:.2f}")
    print(f"  Energy embed:       {energy_norm:.2f}")
    
    # At frames 0-559 (all same phoneme), variance comes ENTIRELY from pitch/energy
    phoneme_var_0_559 = expanded[0, :560].std(dim=0).mean().item()
    pitch_var_0_559 = pitch_emb[0, :560].std(dim=0).mean().item()
    energy_var_0_559 = energy_emb[0, :560].std(dim=0).mean().item()
    
    print(f"\nVariance within frames 0-559 (same phoneme):")
    print(f"  Phoneme std:  {phoneme_var_0_559:.6f} (should be ~0)")
    print(f"  Pitch std:    {pitch_var_0_559:.4f}")
    print(f"  Energy std:   {energy_var_0_559:.4f}")
    
    # Now check the mel prediction
    total_input = model.pos_enc(total_input)
    dec = total_input
    for layer in model.decoder_layers:
        dec = layer(dec)
    mel_pred = model.mel_linear(dec)
    
    mel_norm_gt = (mel_gt - mel_mean) / mel_std
    T = min(mel_pred.size(1), mel_norm_gt.size(1))
    l1 = (mel_pred[:, :T] - mel_norm_gt[:, :T]).abs().mean().item()
    
    # L1 in the "first phoneme" region vs rest
    l1_first = (mel_pred[:, :560] - mel_norm_gt[:, :560]).abs().mean().item()
    l1_rest = (mel_pred[:, 560:T] - mel_norm_gt[:, 560:T]).abs().mean().item()
    
    print(f"\nMel L1 breakdown:")
    print(f"  Overall: {l1:.4f}")
    print(f"  Frames 0-559 (first phoneme region): {l1_first:.4f}")
    print(f"  Frames 560+ (rest): {l1_rest:.4f}")
    
    # Now try with CORRECT durations (rebuild from TextGrid)
    from generate_paddlespeech_distillation_data import (
        parse_textgrid_intervals, text_to_phonemes, build_durations
    )
    tg_path = ROOT / "data" / "mfa_aligned" / f"{pid}.TextGrid"
    intervals = parse_textgrid_intervals(tg_path)
    phonemes = text_to_phonemes(rec["text"])
    correct_durations = build_durations(phonemes, intervals)
    
    # Fix sum to match mel_len
    diff = rec["mel_len"] - sum(correct_durations)
    max_idx = max(range(len(correct_durations)), key=lambda i: correct_durations[i])
    correct_durations[max_idx] = max(correct_durations[max_idx] + diff, 1)
    
    dur_correct = torch.tensor([correct_durations], dtype=torch.long)
    print(f"\nCorrected durations: sum={sum(correct_durations)} mel_len={rec['mel_len']}")
    print(f"First 10: {correct_durations[:10]}")
    print(f"Max single: {max(correct_durations)} at idx {correct_durations.index(max(correct_durations))}")
    
    # Corrected durations sum to 861, but mel has 687 frames.
    # The TextGrid is at a different timescale than the mel.
    # This is a separate bug: TextGrid total=11.92s, mel=687/80=8.59s
    # For now, just compare at the overlap region.
    expanded2 = model.length_regulator(x, dur_correct)
    T2 = min(expanded2.size(1), f0_norm.size(1), e_norm.size(1))
    expanded2 = expanded2[:, :T2]
    pitch_emb2 = model.pitch_embed(f0_norm[:, :T2].unsqueeze(-1))
    energy_emb2 = model.energy_embed(e_norm[:, :T2].unsqueeze(-1))
    input2 = expanded2 + pitch_emb2 + energy_emb2
    input2 = model.pos_enc(input2)
    dec2 = input2
    for layer in model.decoder_layers:
        dec2 = layer(dec2)
    mel_pred2 = model.mel_linear(dec2)
    
    T2_cmp = min(mel_pred2.size(1), mel_norm_gt.size(1))
    l1_corrected = (mel_pred2[:, :T2_cmp] - mel_norm_gt[:, :T2_cmp]).abs().mean().item()
    
    print(f"\nMel L1 with CORRECTED durations (same model, no retrain): {l1_corrected:.4f}")
    print(f"Mel L1 with STORED (garbage) durations:                   {l1:.4f}")
    print(f"Improvement from just fixing durations: {l1 - l1_corrected:.4f}")
