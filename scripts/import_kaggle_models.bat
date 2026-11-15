@echo off
setlocal
cd /d "%~dp0\.."
echo ============================================
echo NOORIX-AI Kaggle Model Import
echo ============================================
if not exist "models\incoming" mkdir "models\incoming"
echo.
echo ZIP sources:
echo  - models\incoming\  (all ZIP files placed here)
echo  - %%USERPROFILE%%\Downloads\  (ZIP files downloaded in the last 2 hours)
echo.
echo The importer keeps original ZIP files and imports only recognized model/config artifacts.
python scripts\import_kaggle_models.py --scan-downloads --since-hours 2
if errorlevel 1 (
  echo.
  echo Import reported an error. Original ZIP files are preserved.
  pause
  exit /b 1
)
echo.
echo Import finished. Review models\imported\ and each import_manifest.json
pause
