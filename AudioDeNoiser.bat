@echo off
setlocal
title Audio DeNoiser
cd /d "%~dp0"

rem  Double-click to run. You can also drop an audio file onto this .bat.
rem  First run builds a private Python 3.11 environment in .venv next to it.

set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PYW=%VENV%\Scripts\pythonw.exe"

if exist "%VENV%\setup-complete.txt" goto launch

echo.
echo  Audio DeNoiser - first-time setup
echo  ---------------------------------
echo  Creating a private Python 3.11 environment in:
echo    %VENV%
echo  and downloading about 300 MB: CPU PyTorch, DeepFilterNet and friends.
echo.

py -3.11 -c "import sys" >nul 2>&1
if errorlevel 1 goto nopython

if exist "%PY%" goto install
echo  Creating the environment...
py -3.11 -m venv "%VENV%"
if errorlevel 1 goto failed

:install
echo  Installing packages, this takes a few minutes...
"%PY%" -m pip install --no-cache-dir --upgrade pip
if errorlevel 1 goto failed
"%PY%" -m pip install --no-cache-dir torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
"%PY%" -m pip install --no-cache-dir deepfilternet==0.5.6 numpy==1.26.4 scipy==1.17.1 soundfile==0.14.0 sounddevice==0.5.6 customtkinter==6.0.0
if errorlevel 1 goto failed

echo done> "%VENV%\setup-complete.txt"
echo.
echo  Setup complete. Starting Audio DeNoiser...

:launch
start "" "%PYW%" "%~dp0audio_denoiser.py" %*
exit /b 0

:nopython
echo  Python 3.11 was not found.
echo.
echo  DeepFilterNet only supports Python 3.8 to 3.11 on Windows, so this app
echo  needs 3.11 even if you already have a newer Python. It installs alongside
echo  your other versions without changing them. Either run:
echo      winget install Python.Python.3.11
echo  or install Python 3.11.9, 64-bit, from python.org. Then run this file again.
echo.
pause
exit /b 1

:failed
echo.
echo  Setup stopped because of the error above. Fix it - usually a dropped
echo  internet connection - and run this file again; it carries on from here.
echo.
pause
exit /b 1
