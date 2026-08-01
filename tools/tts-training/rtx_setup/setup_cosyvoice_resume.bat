@echo off
REM Resume of setup_cosyvoice.bat after STEP 5 failure (openai-whisper sdist
REM build needs pkg_resources, removed from modern setuptools in pip's
REM isolated build env). Steps 1-4 (conda env + pynini + torch) already done.

set PY=C:\Users\wehao\anaconda3\envs\cosyvoice\python.exe

echo [STEP 5a] Pre-installing setuptools^<81 (ships pkg_resources) + wheel
%PY% -m pip install "setuptools<81" wheel -i https://mirrors.aliyun.com/pypi/simple/ || goto :err_5a

echo [STEP 5b] Installing openai-whisper without build isolation
%PY% -m pip install openai-whisper==20231117 --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple/ || goto :err_5b

echo [STEP 5] Installing requirements from E:\rtx_cosyvoice_requirements.txt
%PY% -m pip install -r E:\rtx_cosyvoice_requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ || goto :err_5

echo [STEP 6] Installing WeTextProcessing with --no-deps
%PY% -m pip install WeTextProcessing==1.0.3 --no-deps -i https://mirrors.aliyun.com/pypi/simple/ || goto :err_6

echo [STEP 7] Verifying torch + CosyVoice import
set KMP_DUPLICATE_LIB_OK=TRUE
%PY% -c "import torch; print('TORCH_OK', torch.__version__, torch.cuda.is_available())" || goto :err_7
cd /d E:\CosyVoice || goto :err_7
set COSY_ROOT=E:\CosyVoice
%PY% -c "import sys, os; sys.path.insert(0, os.environ['COSY_ROOT']); sys.path.append(os.path.join(os.environ['COSY_ROOT'], 'third_party', 'Matcha-TTS')); from cosyvoice.cli.cosyvoice import AutoModel; print('IMPORT_OK')" || goto :err_7

echo SETUP_DONE
exit /b 0

:err_5a
echo SETUP_FAILED step=5a code=%errorlevel%
exit /b 51

:err_5b
echo SETUP_FAILED step=5b code=%errorlevel%
exit /b 52

:err_5
echo SETUP_FAILED step=5 code=%errorlevel%
exit /b 5

:err_6
echo SETUP_FAILED step=6 code=%errorlevel%
exit /b 6

:err_7
echo SETUP_FAILED step=7 code=%errorlevel%
exit /b 7
