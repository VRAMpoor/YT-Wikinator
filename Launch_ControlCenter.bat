@echo off
:: Starts the YTC Control Center without a console window and auto-starts
:: the LLM engine + web server. Extra arguments are passed through.
setlocal
cd /d "%~dp0"

set "PYW="
if exist "%~dp0python\pythonw.exe" set "PYW=%~dp0python\pythonw.exe"

:: First pythonw.exe on PATH that really runs and has Tk. A candidate is judged
:: by starting it, not by its folder: a Microsoft Store Python lives under
:: WindowsApps and works, while an alias stub that starts nothing fails here.
if not defined PYW (
    for /f "delims=" %%P in ('where pythonw 2^>nul') do (
        if not defined PYW (
            "%%P" -c "import tkinter" >nul 2>&1
            if not errorlevel 1 set "PYW=%%P"
        )
    )
)

if not defined PYW (
    echo [ERROR] No working pythonw.exe with Tk support was found on PATH.
    echo Install Python 3.10+ from python.org ^(tick "Add python.exe to PATH"^)
    echo or place a Python with pythonw.exe in the "python" subfolder.
    pause
    exit /b 1
)

start "" "%PYW%" "%~dp0YTC_ControlCenter.pyw" --autostart %*
exit /b 0
