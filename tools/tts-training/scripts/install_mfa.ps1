# install_mfa.ps1 — Verify and setup MFA environment
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:PYTHONUNBUFFERED = "1"

Remove-Item "data\mfa_setup.log" -ErrorAction SilentlyContinue

# Find the mfa executable
$mfa_exe = "C:\Users\wsqsy\.conda\envs\mfa\Scripts\mfa.exe"

if (Test-Path $mfa_exe) {
    Write-Output "MFA found at: $mfa_exe"
    & $mfa_exe version 2>&1
    Write-Output "---"
    # Download Mandarin acoustic model and dictionary
    & $mfa_exe model download acoustic mandarin_mfa 2>&1
    Write-Output "---"
    & $mfa_exe model download dictionary mandarin_china_mfa 2>&1
} else {
    Write-Output "MFA not found at expected path. Searching..."
    Get-ChildItem "C:\ProgramData\miniconda3\envs\mfa" -Recurse -Filter "mfa*" -ErrorAction SilentlyContinue | Select-Object FullName
}
