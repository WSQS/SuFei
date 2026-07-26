# mfa_align_plain.ps1 — Full MFA align on plain Chinese character corpus
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:PYTHONUNBUFFERED = "1"
$env:PATH = "C:\Users\wsqsy\.conda\envs\mfa\Library\bin;" + $env:PATH

$mfa_exe = "C:\Users\wsqsy\.conda\envs\mfa\Scripts\mfa.exe"

Remove-Item "data\mfa_align_plain.log" -ErrorAction SilentlyContinue

& $mfa_exe align `
    "data\mfa_corpus_plain" `
    mandarin_china_mfa `
    mandarin_mfa `
    "data\mfa_aligned" `
    --single_speaker `
    --clean `
    --num_jobs 1 `
    2>&1 | Tee-Object -FilePath "data\mfa_align_plain.log"
