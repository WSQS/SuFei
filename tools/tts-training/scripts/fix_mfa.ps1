# fix_mfa.ps1 — Fix soundfile DLL issue and verify MFA
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:PYTHONUNBUFFERED = "1"

$mfa_python = "C:\Users\wsqsy\.conda\envs\mfa\python.exe"
$mfa_exe = "C:\Users\wsqsy\.conda\envs\mfa\Scripts\mfa.exe"

# Fix: reinstall soundfile with proper libs
Write-Output "=== Fixing soundfile ==="
& $mfa_python -m pip install --force-reinstall soundfile 2>&1

# Also try installing libsndfile via conda
Write-Output "=== Installing libsndfile via conda ==="
& conda install -n mfa -c conda-forge libsndfile -y 2>&1

# Test import
Write-Output "=== Testing soundfile import ==="
& $mfa_python -c "import soundfile; print('soundfile OK:', soundfile.__version__)" 2>&1

# Test MFA
Write-Output "=== Testing MFA ==="
& $mfa_exe version 2>&1

# Download models
if (& $mfa_exe version 2>&1) {
    Write-Output "=== Downloading Mandarin acoustic model ==="
    & $mfa_exe model download acoustic mandarin_mfa 2>&1
    Write-Output "=== Downloading Mandarin dictionary ==="
    & $mfa_exe model download dictionary mandarin_china_mfa 2>&1
}

Write-Output "=== Done ==="
