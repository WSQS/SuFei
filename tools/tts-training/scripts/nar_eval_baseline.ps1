# nar_eval_baseline.ps1 — Generate + evaluate NAR baseline on test set
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:PYTHONUNBUFFERED = "1"

Remove-Item "data\nar_baseline.log" -ErrorAction SilentlyContinue

# Generate 24 test poems
& ".\venv_moss\Scripts\python.exe" -u "scripts\nar_generate_testset.py" > "data\nar_baseline.log" 2>&1

# Evaluate
& ".\venv_moss\Scripts\python.exe" -u "scripts/eval_tts.py" `
    --test-set data/test_set_v1.json `
    --audio-dir output/nar_baseline `
    --model-version nar_baseline `
    --teacher-audio-dir data/audio `
    --output-csv data/eval_nar_baseline.csv `
    >> "data\nar_baseline.log" 2>&1
