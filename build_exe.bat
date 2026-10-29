@echo off
rem Reproducible PyInstaller build of AudioDeNoiser (onedir, windowed).
rem Usage: build_exe.bat [extra pyinstaller args, e.g. --console --name AudioDeNoiser-debug]
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "DFMODEL=%LOCALAPPDATA%\DeepFilterNet\DeepFilterNet\Cache\DeepFilterNet3"
if not exist "%DFMODEL%\config.ini" (
  echo Downloading DeepFilterNet3 model into cache...
  "%PY%" -c "from df.enhance import init_df; init_df(log_level='NONE', log_file=None)" || exit /b 1
)
"%PY%" -m pip show pyinstaller >nul 2>&1 || "%PY%" -m pip install pyinstaller || exit /b 1
"%PY%" -m PyInstaller --noconfirm --clean --onedir --windowed ^
  --name AudioDeNoiser --icon icon\audio_denoiser.ico ^
  --version-file packaging\version_info.txt ^
  --add-data "icon;icon" ^
  --add-data "%DFMODEL%;df_models\DeepFilterNet3" ^
  --runtime-hook packaging\rth_df_model.py ^
  --collect-all customtkinter --collect-all df --collect-all libdf ^
  --collect-data torchaudio --collect-binaries torchaudio --hidden-import torchaudio.lib.libtorchaudio ^
  --collect-all soundfile --collect-all sounddevice --hidden-import appdirs ^
  --exclude-module matplotlib --exclude-module IPython --exclude-module pytest --exclude-module tensorboard ^
  %* audio_denoiser.py
exit /b %errorlevel%
