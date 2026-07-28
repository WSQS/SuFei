# Copy NAR TTS models to Android assets
# Run from project root: pwsh scripts/copy_nar_models.ps1

$srcDir = "tools\tts-training\models\paddlespeech_onnx"
$fs2Src = Join-Path $srcDir "fastspeech2_csmsc_onnx_0.2.0\fastspeech2_csmsc.onnx"
$hifiganSrc = Join-Path $srcDir "hifigan_csmsc_onnx_0.2.0\hifigan_csmsc.onnx"
$phoneMapSrc = Join-Path $srcDir "fastspeech2_csmsc_onnx_0.2.0\phone_id_map.txt"

$destDir = "app\src\main\assets\models\nar"
New-Item -ItemType Directory -Path $destDir -Force | Out-Null

Write-Output "Copying NAR TTS models to $destDir..."

# Copy ONNX models (compressed in assets)
Copy-Item $fs2Src (Join-Path $destDir "fastspeech2_csmsc.onnx") -Force
Write-Output "  FS2: $([math]::Round((Get-Item $fs2Src).Length/1MB, 1))MB"

Copy-Item $hifiganSrc (Join-Path $destDir "hifigan_csmsc.onnx") -Force
Write-Output "  HiFiGAN: $([math]::Round((Get-Item $hifiganSrc).Length/1MB, 1))MB"

Copy-Item $phoneMapSrc (Join-Path $destDir "phone_id_map.txt") -Force
Write-Output "  phone_id_map.txt: $((Get-Item $phoneMapSrc).Length) bytes"

$total = (Get-Item $fs2Src).Length + (Get-Item $hifiganSrc).Length
Write-Output "Total: $([math]::Round($total/1MB, 1))MB"
