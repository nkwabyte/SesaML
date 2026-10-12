# Checks if a compatible Python (>=3.10, <3.14) is installed on Windows.
# If none is found, auto-installs Python 3.11 via uv or winget.
#
# Usage:
#   .\scripts\\setup\\ensure_python.ps1

$ErrorActionPreference = "Stop"

function Test-PythonVersion($pythonPath) {
    if (-not (Test-Path $pythonPath -PathType Leaf)) { return $false }
    try {
        $output = & $pythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($output) {
            $parts = $output.Trim().Split('.')
            $major = [int]$parts[0]
            $minor = [int]$parts[1]
            if ($major -eq 3 -and $minor -ge 10 -and $minor -lt 14) {
                return $true
            }
        }
    } catch {
        return $false
    }
    return $false
}

function Find-CompatiblePython {
    if ($env:PYTHON -and (Test-PythonVersion $env:PYTHON)) {
        return $env:PYTHON
    }

    $candidates = @("python3.11", "python3.12", "python3.13", "python3.10", "python", "py")
    foreach ($cmd in $candidates) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1
        if ($found -and (Test-PythonVersion $found)) {
            return $found
        }
    }

    # Check default Windows installation paths
    $defaultPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:ProgramFiles\Python311\python.exe",
        "$env:USERPROFILE\.local\share\uv\python\cpython-3.11*\python.exe"
    )
    foreach ($p in $defaultPaths) {
        $resolved = Resolve-Path $p -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Path -First 1
        if ($resolved -and (Test-PythonVersion $resolved)) {
            return $resolved
        }
    }

    return $null
}

$pyPath = Find-CompatiblePython
if ($pyPath) {
    Write-Output $pyPath
    exit 0
}

Write-Host "[sesaml] No compatible Python (>=3.10, <3.14) found. Attempting to install Python 3.11..." -ForegroundColor Yellow

# 1. Try uv (zero-root / portable installer)
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "[sesaml] Installing uv package manager..." -ForegroundColor Cyan
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        $env:Path += ";$env:USERPROFILE\.cargo\bin;$env:USERPROFILE\.local\bin"
    }
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Host "[sesaml] Installing Python 3.11 via uv..." -ForegroundColor Cyan
        uv python install 3.11
        $uvPy = (uv python find 3.11 2>$null)
        if ($uvPy -and (Test-PythonVersion $uvPy)) {
            Write-Output $uvPy
            exit 0
        }
    }
} catch {
    # Fallback to winget if uv fails
}

# 2. Try winget (Windows Package Manager)
if (Get-Command winget -ErrorAction SilentlyContinue) {
    try {
        Write-Host "[sesaml] Installing Python 3.11 via winget..." -ForegroundColor Cyan
        winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements
        $pyPath = Find-CompatiblePython
        if ($pyPath) {
            Write-Output $pyPath
            exit 0
        }
    } catch {}
}

Write-Error "[sesaml] Failed to automatically install Python 3.11. Please download and install Python 3.11 manually from https://python.org."
exit 1
