$env:JAVA_HOME = "C:\Users\wsqsy\.jdks\ms-17.0.17"
& "tools\tts-training\venv_moss\Scripts\python.exe" "tools\tts-training\scripts\nar_train.py" `
    --steps 10000 `
    --batch_size 8 `
    --lr 5e-4 `
    --warmup_steps 500 `
    --log_interval 100 `
    --save_interval 2000 `
    --resume "tools\tts-training\checkpoints\fs2_step6000.pt" `
    --w_dur 1.0 `
    --w_pitch 0.5 `
    --w_energy 0.5
