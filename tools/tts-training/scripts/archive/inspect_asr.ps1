# inspect_asr.ps1
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:PYTHONUNBUFFERED = "1"
Remove-Item "data\inspect_asr.log" -ErrorAction SilentlyContinue
& ".\venv_moss\Scripts\python.exe" -u "scripts\inspect_asr.py" > "data\inspect_asr.log" 2>&1
