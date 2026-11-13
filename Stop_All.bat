@echo off
title Stopping YTC Services
cd /d "%~dp0"

echo ===========================================================================
echo   Stopping YTC web server and LLM engine (ports from config.json)...
echo ===========================================================================

:: Only the ytc.exe / llama-server.exe that listen on the YTC ports are stopped.
:: Other llama-server instances on this PC are left alone.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$found = $false; $yp = 8088; $lp = 8080; try { $c = Get-Content -Raw -LiteralPath config.json -ErrorAction Stop | ConvertFrom-Json; if ($c.ytc.port) { $yp = [int]$c.ytc.port }; if ($c.llm.port) { $lp = [int]$c.llm.port } } catch { }; foreach ($port in $yp, $lp) { Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { $p = Get-Process -Id $_ -ErrorAction SilentlyContinue; if ($p -and ($p.ProcessName -eq 'ytc' -or $p.ProcessName -eq 'llama-server')) { taskkill /PID $p.Id /T /F | Out-Null; Write-Host ('[*] Stopped ' + $p.ProcessName + ' (PID ' + $p.Id + ') on port ' + $port); $found = $true } elseif ($p) { Write-Host ('[!] Port ' + $port + ' belongs to ' + $p.ProcessName + ' - left alone') } } }; if (-not $found) { Write-Host '[*] No YTC services were running.' }"

echo ===========================================================================
echo   Done.
echo ===========================================================================
timeout /t 3 /nobreak >nul
