# TTS Training — Known Issues & TODO

Issues discovered during experimentation. To be fixed after current H1/H2 evaluation round completes.

---

## 1. Comma (，) encoded as `<unk>` instead of pid=263

**Status**: FIXED

**Root cause**: `PUNCT_TO_PHONE` in `generate_paddlespeech_distillation_data.py:58` mapped `，` (U+FF0C) to two U+FFFD replacement characters (encoding corruption during file creation) instead of the actual `，` that exists in `phone_id_map.txt` (pid=263).

**Fix applied**:
- Corrected the mojibake value in `PUNCT_TO_PHONE`
- Ran `fix_comma_manifests.py` to regenerate `phoneme_ids` across all 5 manifests (394 records, 0 length mismatches)
- Verified: 702 commas now map to pid=263, pid=1 (`<unk>`) count dropped to 0
- Durations/mel/f0/energy unchanged — only phoneme_ids were regenerated

**Note**: All H1/H2 experiments were run with the buggy manifests. Results remain valid for relative comparisons. D300+ experiments will use corrected manifests.

---

## 2. H1 mel_loss higher than E3 at same step count

**Status**: Observation, not a bug

H1 (171 poems) reaches mel_loss=0.171 at 24k steps. E3 (191 poems) reached mel_loss=0.175 at 24k steps. H1 is slightly lower, which is expected (fewer training sequences = faster fitting). No action needed.

---

## 3. copyparty checkpoint downloads truncated/unreliable

**Status**: Infrastructure issue

copyparty on rtx serves files but large downloads (>35MB) frequently stall or truncate over WiFi (~200KB/s). SCP also times out on large files.

**Workaround**: Strip optimizer/scheduler state to reduce checkpoint size (83MB → 35MB). Still slow.

**Future fix options**:
- Deploy a more robust HTTP server (e.g., `python -m http.server` with chunked encoding)
- Use `rsync` over SSH with `--partial --append-verify`
- Zip checkpoints before transfer
- Investigate WiFi bandwidth/interference issue

---

## 4. Windows .bat files uploaded via copyparty lose CRLF encoding

**Status**: Workaround in place

Files written on Windows with LF-only line endings cause `cmd.exe` to fail silently when uploaded to rtx.

**Fix**: Always convert to CRLF before upload (`[System.IO.File]::WriteAllBytes` with `\r\n` replacement), or use SCP instead of copyparty for .bat files.

---

## 5. Start-Process via SSH doesn't persist after session close

**Status**: Workaround in place

`Start-Process -WindowStyle Hidden` launched via SSH is killed when the SSH session ends. Scheduled tasks (`Register-ScheduledTask` with `LogonType Interactive`) work reliably as a workaround.

**Future fix**: Create a persistent service or use `nssm` to install copyparty/training as a Windows service.
