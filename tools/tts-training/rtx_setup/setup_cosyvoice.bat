@echo off
REM ============================================================================
REM RTX CosyVoice environment setup (native Windows conda, no WSL)
REM scp'd to E:\ on rtx, run via ssh under cmd.
REM ============================================================================

set CONDA=C:\Users\wehao\anaconda3\condabin\conda.bat
set ENV=cosyvoice
set PY=C:\Users\wehao\anaconda3\envs\cosyvoice\python.exe

echo [STEP 1] Removing existing conda env (if any) and creating fresh one
call %CONDA% env remove -n %ENV% -y || ver>nul
call %CONDA% create -n %ENV% python=3.10 -y || goto :err_1

echo [STEP 2] Installing pynini from conda-forge
call %CONDA% install -n %ENV% -c conda-forge pynini=2.1.5 -y || goto :err_2

echo [STEP 3] Setting python path
REM PY already set above; verify it exists
if not exist "%PY%" goto :err_3

echo [STEP 4] Installing torch+torchaudio (Aliyun mirror, official fallback)
%PY% -m pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 -f https://mirrors.aliyun.com/pytorch-wheels/cu124/ || %PY% -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 || goto :err_4

echo [STEP 5a] Pre-installing setuptools^<81 (ships pkg_resources) + wheel
REM openai-whisper's sdist setup.py imports pkg_resources at build time,
REM which modern setuptools (81+) in pip's isolated build env no longer ships.
%PY% -m pip install "setuptools<81" wheel -i https://mirrors.aliyun.com/pypi/simple/ || goto :err_5

echo [STEP 5b] Installing openai-whisper without build isolation
%PY% -m pip install openai-whisper==20231117 --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple/ || goto :err_5

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

:err_1
echo SETUP_FAILED step=1 code=%errorlevel%
exit /b 1

:err_2
echo SETUP_FAILED step=2 code=%errorlevel%
exit /b 2

:err_3
echo SETUP_FAILED step=3 code=%errorlevel%
exit /b 3

:err_4
echo SETUP_FAILED step=4 code=%errorlevel%
exit /b 4

:err_5
echo SETUP_FAILED step=5 code=%errorlevel%
exit /b 5

:err_6
echo SETUP_FAILED step=6 code=%errorlevel%
exit /b 6

:err_7
echo SETUP_FAILED step=7 code=%errorlevel%
exit /b 7
