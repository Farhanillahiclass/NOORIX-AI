@echo off
setlocal
cd /d "%~dp0\.."
echo ============================================
echo NOORIX-AI Kaggle Model Import
echo ============================================
if not exist "models\incoming" mkdir "models\incoming"
echo.
echo 1. Copy your Kaggle ZIP files into:
echo    %CD%\models\incoming
echo.
echo 2. This tool extracts only recognized model weights and config files.
echo 3. Temporary extracted files that are not needed are discarded.
echo 4. Original ZIP files are kept for safety.
echo.
python scripts\import_kaggle_models.py
if errorlevel 1 (
  echo.
  echo Import reported an error. Original ZIP files are preserved.
  pause
  exit /b 1
)
echo.
echo Import finished. Check models\imported\
pause
