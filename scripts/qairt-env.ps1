#Requires -Version 5.1
<#
.SYNOPSIS
    QAIRT SDK environment helper for Windows (PowerShell).

.DESCRIPTION
    Dot-source this script to set up the environment for QAIRT model conversion
    (qnn-onnx-converter, qnn-model-lib-generator, qnn-context-binary-generator)
    on Windows.

        . .\scripts\qairt-env.ps1
        $env:QAIRT_SDK_ROOT = "D:\Qualcomm\AIStack\QAIRT\2.31.0.250130"
        . .\scripts\qairt-env.ps1

    It will:
      - default QAIRT_SDK_ROOT to C:\Qualcomm\AIStack\QAIRT\2.31.0.250130 if it
        exists, otherwise $HOME\Qualcomm\AIStack\QAIRT\2.31.0.250130, and
        ANDROID_NDK_ROOT to C:\android_sdk\ndk\26.1.10909125 if it exists,
        otherwise $HOME\Android\Sdk\ndk\26.1.10909125
      - create .venv-qairt (Python 3.10) with the converter dependencies
      - add the SDK python dir to PYTHONPATH
      - locate the converter bin directory and add it to PATH
      - validate that the converter executables are reachable

.NOTES
    This mirrors scripts/qairt-env.sh (the Linux helper). The Windows SDK's bin
    directory name varies between QAIRT releases, so this script *searches* for
    the directory that contains qnn-onnx-converter instead of hardcoding it.
    If your install still is not found, run the SDK's own environment setup
    (or a Visual Studio developer prompt) first, then re-run this script.
#>

[CmdletBinding()]
param(
    [string]$SdkRoot = $env:QAIRT_SDK_ROOT,
    [string]$NdkRoot = $env:ANDROID_NDK_ROOT
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# ---- QAIRT_SDK_ROOT --------------------------------------------------------
# Windows installers put the SDK at C:\Qualcomm\...; some hosts keep it under
# $HOME\Qualcomm\.... Prefer whichever exists on this machine, and drop a value
# that is set but points nowhere so a stale inherited variable cannot shadow the
# default and trip the check below.
if ($SdkRoot -and -not (Test-Path -LiteralPath $SdkRoot)) {
    Write-Warning "[qairt-env] Ignoring QAIRT_SDK_ROOT='$SdkRoot' (not a directory)."
    $SdkRoot = $null
}
if (-not $SdkRoot) {
    $WindowsRoot = "C:\Qualcomm\AIStack\QAIRT\2.31.0.250130"
    $SdkRoot = if (Test-Path -LiteralPath $WindowsRoot) {
        $WindowsRoot
    }
    else {
        Join-Path $HOME "Qualcomm\AIStack\QAIRT\2.31.0.250130"
    }
}
if (-not (Test-Path -LiteralPath $SdkRoot)) {
    throw "[qairt-env] QAIRT_SDK_ROOT does not exist: $SdkRoot"
}
$env:QAIRT_SDK_ROOT = (Resolve-Path -LiteralPath $SdkRoot).Path

# ---- ANDROID_NDK_ROOT (optional) -------------------------------------------
# Only needed by the qnn-model-lib / context-binary steps; same fallback logic
# as QAIRT_SDK_ROOT above.
if ($NdkRoot -and -not (Test-Path -LiteralPath $NdkRoot)) {
    Write-Warning "[qairt-env] Ignoring ANDROID_NDK_ROOT='$NdkRoot' (not a directory)."
    $NdkRoot = $null
}
if (-not $NdkRoot) {
    $NdkCandidates = @(
        "C:\android_sdk\ndk\26.1.10909125",
        (Join-Path $HOME "Android\Sdk\ndk\26.1.10909125")
    )
    foreach ($candidate in $NdkCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            $NdkRoot = $candidate
            break
        }
    }
}
if ($NdkRoot) {
    $env:ANDROID_NDK_ROOT = (Resolve-Path -LiteralPath $NdkRoot).Path
}

# ---- Converter venv (Python 3.10) -----------------------------------------
# The SDK's compiled modules link against Python 3.10; the project venv is 3.12.
$VenvDir = Join-Path $RepoRoot ".venv-qairt"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "[qairt-env] uv is not installed. Install it: irm https://astral.sh/uv/install.ps1 | iex"
    }
    Write-Host "[qairt-env] Creating converter venv at $VenvDir ..."
    uv venv --python 3.10 $VenvDir
    uv pip install --python $VenvPython `
        --index-strategy unsafe-best-match `
        onnx==1.16.1 onnxruntime==1.17.1 "numpy<2" onnx-simplifier `
        scipy lxml absl-py pandas pyyaml
}
$env:VIRTUAL_ENV = $VenvDir
$env:PATH = (Join-Path $VenvDir "Scripts") + ";" + $env:PATH
Write-Host "[qairt-env] Activated venv: $VenvDir"

# ---- SDK python packages ---------------------------------------------------
$SdkPython = Join-Path $env:QAIRT_SDK_ROOT "lib\python"
if (Test-Path -LiteralPath $SdkPython) {
    $env:PYTHONPATH = if ($env:PYTHONPATH) { "$SdkPython;$env:PYTHONPATH" } else { $SdkPython }
}

# ---- Locate the converter bin directory ------------------------------------
$Converters = @(
    "qnn-onnx-converter",
    "qnn-model-lib-generator",
    "qnn-context-binary-generator"
)

$BinDir = $null
$BinRoot = Join-Path $env:QAIRT_SDK_ROOT "bin"
if (Test-Path -LiteralPath $BinRoot) {
    foreach ($candidate in Get-ChildItem -LiteralPath $BinRoot -Directory -ErrorAction SilentlyContinue) {
        if (Get-ChildItem -LiteralPath $candidate.FullName -Filter "qnn-onnx-converter*" -ErrorAction SilentlyContinue) {
            $BinDir = $candidate.FullName
            break
        }
    }
    if (-not $BinDir -and (Get-ChildItem -LiteralPath $BinRoot -Filter "qnn-onnx-converter*" -ErrorAction SilentlyContinue)) {
        $BinDir = $BinRoot
    }
}
if ($BinDir) { $env:PATH = $BinDir + ";" + $env:PATH }

# ---- Verification ----------------------------------------------------------
Write-Host ""
Write-Host "============================================"
Write-Host " QAIRT Environment - Verification"
Write-Host "============================================"
Write-Host " QAIRT_SDK_ROOT  : $env:QAIRT_SDK_ROOT"
Write-Host " ANDROID_NDK_ROOT: $($env:ANDROID_NDK_ROOT)"
Write-Host " Converter venv  : $VenvDir"
Write-Host " Converter bin   : $(if ($BinDir) { $BinDir } else { '<not found>' })"

$allFound = $true
foreach ($tool in $Converters) {
    $cmd = Get-Command $tool -ErrorAction SilentlyContinue
    if ($cmd) {
        Write-Host (" {0,31} : {1}  OK" -f $tool, $cmd.Source)
    }
    else {
        Write-Host (" {0,31} : NOT FOUND" -f $tool)
        $allFound = $false
    }
}

if ($allFound) {
    Write-Host "[qairt-env] All converters reachable."
}
else {
    Write-Warning "[qairt-env] Some converters were not found. Run the SDK's own environment setup (or a VS developer prompt) first, then re-run this script."
}
