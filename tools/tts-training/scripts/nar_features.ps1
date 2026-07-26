# nar_features.ps1 — Extract Mel/F0/energy features
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:PYTHONUNBUFFERED = "1"
Remove-Item "data\nar_features.log" -ErrorAction SilentlyContinue
& ".\venv_moss\Scripts\python.exe" -u "scripts\nar_extract_features.py" > "data\nar_features.log" 2>&1
