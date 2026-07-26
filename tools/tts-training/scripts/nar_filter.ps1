# nar_filter.ps1
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:PYTHONUNBUFFERED = "1"
Remove-Item "data\nar_filter.log" -ErrorAction SilentlyContinue
& ".\venv_moss\Scripts\python.exe" -u "scripts\nar_filter_teacher.py" > "data\nar_filter.log" 2>&1
