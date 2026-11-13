# Creates (or refreshes) the "YTC Control Center" icon on the Desktop.
# Run again if you move the project folder or reinstall Python:
#   powershell -ExecutionPolicy Bypass -File Create_Desktop_Shortcut.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$app  = Join-Path $root 'YTC_ControlCenter.pyw'
$icon = Join-Path $root 'ytc.ico'

# pythonw.exe = Python without a console window. Prefer the one in the "python" subfolder, then the
# first one on PATH that really runs and has Tk. A candidate is judged by
# starting it, not by its folder: a Microsoft Store Python lives under
# WindowsApps and works, while an alias stub that starts nothing fails here.
function Test-Pythonw([string]$exe) {
    try {
        $p = Start-Process -FilePath $exe -ArgumentList '-c "import tkinter"' -Wait -PassThru -WindowStyle Hidden
        return ($p.ExitCode -eq 0)
    } catch {
        return $false
    }
}

$pythonw = Join-Path $root 'python\pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonw)) {
    $pythonw = $null
    $found = @(Get-Command pythonw.exe -All -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
    foreach ($candidate in $found) {
        if (Test-Pythonw $candidate) { $pythonw = $candidate; break }
    }
}
if (-not $pythonw) { throw 'No working pythonw.exe with Tk support was found. Install Python 3.10+ and add it to PATH.' }
if (-not (Test-Path -LiteralPath $app)) { throw "Not found: $app" }

$desktop = [Environment]::GetFolderPath('Desktop')
$lnkPath = Join-Path $desktop 'YTC Control Center.lnk'

$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($lnkPath)
$lnk.TargetPath       = $pythonw
$lnk.Arguments        = "`"$app`" --autostart"
$lnk.WorkingDirectory = $root
$lnk.Description      = 'Start the YTC Control Center, LLM engine and wiki web server'
if (Test-Path -LiteralPath $icon) { $lnk.IconLocation = "$icon,0" }
$lnk.Save()

Write-Host "Created: $lnkPath"
Write-Host "  Target: $pythonw $($lnk.Arguments)"
