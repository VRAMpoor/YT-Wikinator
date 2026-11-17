@echo off
title Stopping YTC Services
cd /d "%~dp0"

echo ===========================================================================
echo   Stopping YTC web server and LLM engine (ports from config.json)...
echo ===========================================================================

:: Only the ytc.exe / llama-server.exe that listen on the YTC ports are stopped.
:: Other llama-server instances on this PC are left alone.
:: Before each stop it leaves logs\stop.request: an open Control Center then
:: reports the stop as requested, not as a crash, and offers no restart.
:: Before anything is stopped: is work running that the stop would cut off?
::   - a job of the web server (crawl, discover, re-analysis ...): GET /api/job
::   - a crawl of any program on this library: it holds data\crawl.lock open
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$found = $false; $yp = 8088; $lp = 8080; $db = 'data/ytc.db'; try { $c = Get-Content -Raw -LiteralPath config.json -ErrorAction Stop | ConvertFrom-Json; if ($c.ytc.port) { $yp = [int]$c.ytc.port }; if ($c.llm.port) { $lp = [int]$c.llm.port }; if ($c.ytc.db_path) { $db = [string]$c.ytc.db_path } } catch { }; $work = @(); try { $j = Invoke-RestMethod -Uri ('http://127.0.0.1:' + $yp + '/api/job') -TimeoutSec 2; if ($j.running) { $work += ('the web server runs ' + $j.running) } } catch { }; try { $lock = Join-Path (Split-Path -Parent ([IO.Path]::GetFullPath($db))) 'crawl.lock'; if (Test-Path -LiteralPath $lock) { try { $h = [IO.File]::Open($lock, 'Open', 'ReadWrite', 'None'); $h.Close() } catch [IO.IOException] { $work += 'a crawl is running on this library' } } } catch { }; if ($work.Count -gt 0) { Write-Host ''; Write-Host '[!] Work is still running:'; $work | ForEach-Object { Write-Host ('      - ' + $_) }; Write-Host '    Stopping now cuts it off. Videos that are not finished are not judged.'; Write-Host ''; $a = Read-Host '    Type Y and press Enter to stop anyway (anything else keeps it running)'; if ($a -ne 'Y' -and $a -ne 'y') { Write-Host '[*] Nothing was stopped.'; exit 0 } }; foreach ($port in $yp, $lp) { Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { $p = Get-Process -Id $_ -ErrorAction SilentlyContinue; if ($p -and ($p.ProcessName -eq 'ytc' -or $p.ProcessName -eq 'llama-server')) { try { New-Item -ItemType Directory -Force -Path logs | Out-Null; Set-Content -LiteralPath 'logs\stop.request' -Value 'Stop_All.bat' -ErrorAction Stop } catch { }; taskkill /PID $p.Id /T /F | Out-Null; Write-Host ('[*] Stopped ' + $p.ProcessName + ' (PID ' + $p.Id + ') on port ' + $port); $found = $true } elseif ($p) { Write-Host ('[!] Port ' + $port + ' belongs to ' + $p.ProcessName + ' - left alone') } } }; if (-not $found) { Write-Host '[*] No YTC services were running.' }"

echo ===========================================================================
echo   Done.
echo ===========================================================================
timeout /t 3 /nobreak >nul
