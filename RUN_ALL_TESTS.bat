@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo       NOORIX AI - ONE COMMAND TEST
echo ==========================================
echo.

if exist "D:\NOORIX_MODELS\EXTRACTED\models" (
    set "NOORIX_MODELS_DIR=D:\NOORIX_MODELS\EXTRACTED"
    echo Found extracted model package: D:\NOORIX_MODELS\EXTRACTED
)

set "CONFIG=config.json"
if not exist "%CONFIG%" (
    set "CONFIG=config.example.json"
    echo Using config.example.json. Add config.json for custom settings.
)

REM Prefer the active Python from PATH (for example, an activated Conda env).
REM Use the project .venv only when no Python command is available.
set "PYTHON=python"
where python >nul 2>nul
if errorlevel 1 (
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON=.venv\Scripts\python.exe"
    ) else (
        where py >nul 2>nul
        if not errorlevel 1 set "PYTHON=py -3.12"
    )
)

echo Using Python:
%PYTHON% -c "import sys; print(sys.executable)"
%PYTHON% "scripts\test_all_models.py" --config "%CONFIG%"
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
