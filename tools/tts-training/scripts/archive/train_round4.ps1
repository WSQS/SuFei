$env:JAVA_HOME = "C:\Users\wsqsy\.jdks\ms-17.0.17"
& "tools\tts-training\venv_moss\Scripts\python.exe" "tools\tts-training\scripts\nar_train.py" `
    --steps 15000 `
    --batch_size 8 `
    --lr 3e-4 `
    --warmup_steps 300 `
    --log_interval 100 `
    --save_interval 2000 `
    --resume "tools\tts-training\checkpoints\fs2_step10000.pt" `
    --reset_dur_bias `
    --w_dur 5.0 `
    --w_mel 1.0 `
    --w_pitch 0.3 `
    --w_energy 0.3
