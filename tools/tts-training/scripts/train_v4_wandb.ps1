# train_v4_wandb.ps1 — SFT training max_length=8192 with wandb enabled
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"

$env:JAVA_HOME = "C:\Users\wsqsy\.jdks\ms-17.0.17"
$env:PYTHONUNBUFFERED = "1"

Remove-Item "data\train_v4_wandb.log" -ErrorAction SilentlyContinue

& ".\venv_moss\Scripts\python.exe" -u `
    ".\MOSS-TTS-Nano\finetuning\sft.py" `
    --model-path models/MOSS-TTS-Nano `
    --codec-path models/MOSS-Audio-Tokenizer-Nano `
    --train-jsonl data/train_with_codes.jsonl `
    --output-dir output/moss_poetry_sft_320_v4_wandb `
    --per-device-batch-size 1 `
    --gradient-accumulation-steps 8 `
    --learning-rate 1e-5 `
    --num-epochs 3 `
    --mixed-precision bf16 `
    --max-length 8192 `
    --channelwise-loss-weight 1,32 `
    > "data\train_v4_wandb.log" 2>&1
