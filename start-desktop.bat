@echo off
setlocal
cd /d "%~dp0"
if exist "runtime\python.exe" (
    "runtime\python.exe" run.py %*
) else (
    python run.py %*
)
if errorlevel 1 (
    echo.
    echo [AI-RVC Studio] 启动失败。请检查 Python / requirements_desktop.txt。
    pause
)
