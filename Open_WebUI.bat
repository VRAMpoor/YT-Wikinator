@echo off
:: Opens the web UI in the default browser, on the port from config.json
:: (ytc.port). Falls back to 8088 when config.json cannot be read.
setlocal
cd /d "%~dp0"
set "YTC_PORT=8088"
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $p = [int](Get-Content -Raw -LiteralPath config.json -ErrorAction Stop | ConvertFrom-Json).ytc.port; if ($p -ge 1 -and $p -le 65535) { $p } } catch { }" 2^>nul`) do set "YTC_PORT=%%P"
start "" "http://localhost:%YTC_PORT%"
