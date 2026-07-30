# eval_v4.ps1 — Evaluate MOSS v4 against test set
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:PYTHONUNBUFFERED = "1"

Remove-Item "data\eval_v4.log" -ErrorAction SilentlyContinue

& ".\venv_moss\Scripts\python.exe" -u "scripts\eval_tts.py" `
    --test-set data/test_set_v1.json `
    --audio-dir output/test_audio_v4 `
    --model-version v4 `
    --teacher-audio-dir data/audio `
    --output-csv data/eval_v4.csv `
    > "data\eval_v4.log" 2>&1
