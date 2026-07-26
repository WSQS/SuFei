# eval_v4_greedy.ps1
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:PYTHONUNBUFFERED = "1"

Remove-Item "data\gen_test_greedy.log" -ErrorAction SilentlyContinue

& ".\venv_moss\Scripts\python.exe" -u "scripts\generate_test_audio.py" > "data\gen_test_greedy.log" 2>&1

Remove-Item "data\eval_v4_greedy.log" -ErrorAction SilentlyContinue
& ".\venv_moss\Scripts\python.exe" -u "scripts\eval_tts.py" `
    --test-set data/test_set_v1.json `
    --audio-dir output/test_audio_v4_greedy `
    --model-version v4_greedy `
    --teacher-audio-dir data/audio `
    --output-csv data/eval_v4_greedy.csv `
    > "data\eval_v4_greedy.log" 2>&1
