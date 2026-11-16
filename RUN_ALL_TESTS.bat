@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo       NOORIX AI - ONE COMMAND TEST
echo ==========================================
echo.

if not exist "config.json" (
    echo ERROR: config.json was not found.
    echo First copy config.example.json to config.json
    echo Then confirm model paths match your extracted model folders.
    echo.
    pause
    exit /b 2
)

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

%PYTHON% "scripts\test_all_models.py" --config "config.json"
set "RESULT=%ERRORLEVEL%"

echo.
if "%RESULT%"=="0" (
    echo Test sequence finished. Review reports\model-tests.
) else (
    echo Test sequence reported a failure. Review the report above.
)
echo.
pause
exit /b %RESULT%
