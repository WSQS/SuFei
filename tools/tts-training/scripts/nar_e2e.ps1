# nar_e2e.ps1
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:PYTHONUNBUFFERED = "1"
Remove-Item "data\nar_e2e.log" -ErrorAction SilentlyContinue
& ".\venv_moss\Scripts\python.exe" -u "scripts\nar_stage1_e2e.py" > "data\nar_e2e.log" 2>&1
