# generate_test_audio.ps1 — Generate 24 test poems with MOSS v4
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:WANDB_DISABLED = "true"
$env:JAVA_HOME = "C:\Users\wsqsy\.jdks\ms-17.0.17"
$env:PYTHONUNBUFFERED = "1"

Remove-Item "data\gen_test.log" -ErrorAction SilentlyContinue

& ".\venv_moss\Scripts\python.exe" -u "scripts\generate_test_audio.py" > "data\gen_test.log" 2>&1
