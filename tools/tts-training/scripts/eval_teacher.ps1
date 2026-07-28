# eval_teacher.ps1 — Run teacher audio evaluation to calibrate ASR baseline
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:PYTHONUNBUFFERED = "1"

Remove-Item "data\eval_teacher.log" -ErrorAction SilentlyContinue

& ".\venv_moss\Scripts\python.exe" -u "scripts\eval_tts.py" `
    --test-set data/test_set_v1.json `
    --audio-dir data/audio `
    --model-version teacher `
    --teacher-audio-dir data/audio `
    --output-csv data/eval_teacher.csv `
    > "data\eval_teacher.log" 2>&1
