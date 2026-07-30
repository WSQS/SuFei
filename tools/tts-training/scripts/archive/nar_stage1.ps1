# nar_stage1.ps1
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:PYTHONUNBUFFERED = "1"

Remove-Item "data\nar_stage1.log" -ErrorAction SilentlyContinue

& ".\venv_moss\Scripts\python.exe" -u "scripts\nar_stage1_onnx_probe.py" > "data\nar_stage1.log" 2>&1
