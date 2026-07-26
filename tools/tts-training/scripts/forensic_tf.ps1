# forensic_tf.ps1
Set-Location "C:\Users\wsqsy\Documents\android\SuFei\tools\tts-training"
$env:WANDB_DISABLED = "true"
$env:PYTHONUNBUFFERED = "1"
Remove-Item "data\forensic_tf.log" -ErrorAction SilentlyContinue
& ".\venv_moss\Scripts\python.exe" -u "scripts\forensic_tf_argmax.py" > "data\forensic_tf.log" 2>&1
