#Requires -Version 5.1
<#
.SYNOPSIS
    QAIRT environment helper for Windows (PowerShell adapter).

.DESCRIPTION
    All path and platform resolution lives in scripts/env.py — the single source
    of truth for the SDK root, the converter bin dir, the host library dir, the
    SDK envsetup flavour and the Python 3.10 converter venv. This script only
    creates the venv, imports the resolved environment and verifies it.

        . .\scripts\qairt-env.ps1
        $env:QAIRT_SDK_ROOT = "D:\Qualcomm\AIStack\QAIRT\2.31.0.250130"
        . .\scripts\qairt-env.ps1

.PARAMETER SdkRoot
    QAIRT SDK root; defaults to $env:QAIRT_SDK_ROOT, and is probed when unset.

.PARAMETER NdkRoot
    Android NDK root; defaults to $env:ANDROID_NDK_ROOT, and is probed when unset.

.NOTES
    Mirrors scripts\qairt-env.sh (the Linux/WSL helper), which refuses to run
    under Git Bash for the same reason this one exists: the platform decides the
    converter triple, the venv layout and the envsetup flavour. VIRTUAL_ENV and
    PATH come from the resolver rather than from Scripts\Activate.ps1, which
    would need an ExecutionPolicy exception.

    Requires uv: https://docs.astral.sh/uv/
#>

[CmdletBinding()]
param(
    [string]$SdkRoot = $env:QAIRT_SDK_ROOT,
    [string]$NdkRoot = $env:ANDROID_NDK_ROOT
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ResolverPath = Join-Path $RepoRoot "scripts\env.py"
Push-Location $RepoRoot
try {
    if ($SdkRoot) { $env:QAIRT_SDK_ROOT = $SdkRoot }
    if ($NdkRoot) { $env:ANDROID_NDK_ROOT = $NdkRoot }

    # Prefer a bare interpreter: the resolver is stdlib + PyYAML, so `uv run`
    # would sync the whole project environment just to resolve paths. Escalate
    # when that interpreter cannot read .kavi.yaml, since probing needs the
    # pinned versions. The WindowsApps path is the Microsoft Store stub for
    # `python`, which is not a real interpreter.
    $envPy = @("python", "py") |
        ForEach-Object { Get-Command $_ -ErrorAction SilentlyContinue } |
        Where-Object { $_.Source -and $_.Source -notlike "*\WindowsApps\*" } |
        Select-Object -First 1
    $useBare = $false
    if ($envPy) {
        & $envPy.Source -c "import yaml" 2>$null
        $useBare = $LASTEXITCODE -eq 0
    }
    if ($useBare) {
        $resolverExe = $envPy.Source
        $resolverArgs = @($ResolverPath)
    }
    else {
        Write-Host "[qairt-env] No local Python with PyYAML; using 'uv run' to resolve paths."
        $resolverExe = "uv"
        $resolverArgs = @("run", "--project", $RepoRoot, "python", $ResolverPath)
    }

    # Create the venv if needed, then verify: abort before exporting anything if
    # the SDK, NDK or converters are not where the platform says they should be.
    & $resolverExe @resolverArgs --bootstrap --check
    if ($LASTEXITCODE -ne 0) {
        throw "[qairt-env] Environment check failed - fix the problem(s) above and retry."
    }

    $payload = & $resolverExe @resolverArgs --format powershell
    if ($LASTEXITCODE -ne 0) {
        throw "[qairt-env] Could not resolve the environment (scripts/env.py failed)."
    }
    Invoke-Expression $payload

    # The vendor SDK's own PowerShell envsetup sets QNN_SDK_ROOT/SNPE_ROOT.
    if (Test-Path -LiteralPath $env:KAVI_ENVSETUP) {
        try {
            . $env:KAVI_ENVSETUP
        }
        catch {
            Write-Warning "[qairt-env] Could not dot-source $env:KAVI_ENVSETUP : $_"
        }
    }
    else {
        Write-Warning "[qairt-env] $($env:KAVI_ENVSETUP) not found - QNN_SDK_ROOT/SNPE_ROOT may be unset."
    }

    Write-Host "[qairt-env] Ready: python from $env:VIRTUAL_ENV"
}
finally {
    Pop-Location
}
