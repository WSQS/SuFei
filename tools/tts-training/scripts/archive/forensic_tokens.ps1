# forensic_tokens.ps1
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:WANDB_DISABLED = "true"
$env:PYTHONUNBUFFERED = "1"
Remove-Item "data\forensic_tokens.log" -ErrorAction SilentlyContinue
& ".\venv_moss\Scripts\python.exe" -u "scripts\forensic_tokens.py" > "data\forensic_tokens.log" 2>&1
