$env:JAVA_HOME = "C:\Users\wsqsy\.jdks\ms-17.0.17"
& "tools\tts-training\venv_moss\Scripts\python.exe" "tools\tts-training\scripts\nar_train.py" `
    --steps 10000 `
    --batch_size 8 `
    --lr 5e-4 `
    --warmup_steps 500 `
    --log_interval 100 `
    --save_interval 2000 `
    --w_dur 5.0 `
    --w_mel 1.0 `
    --w_pitch 2.0 `
    --w_energy 2.0 `
    --wandb_project "sufei-tts" `
    --wandb_name "round6-pred-pitch-energy"