#!/usr/bin/env bash
# shellcheck disable=SC1091
#
# qairt-env.sh — QAIRT SDK environment helper
#
# Source this script to set up the environment for QAIRT 2.31.0.250130
# model conversion (qnn-onnx-converter, qnn-model-lib-generator,
# qnn-context-binary-generator).
#
# Usage:
#   source scripts/qairt-env.sh
#   QAIRT_SDK_ROOT=/custom/path source scripts/qairt-env.sh
#
# This script:
#   - Defaults QAIRT_SDK_ROOT to C:\Qualcomm\AIStack\QAIRT\2.31.0.250130 (Windows)
#     or $HOME/Qualcomm/AIStack/QAIRT/2.31.0.250130, and ANDROID_NDK_ROOT to
#     C:\android_sdk\ndk\26.1.10909125 or $HOME/Android/Sdk/ndk/26.1.10909125
#   - Creates .venv-qairt/ in the repo if missing (Python 3.10, qairt deps)
#   - Sources $QAIRT_SDK_ROOT/bin/envsetup.sh (sets QNN_SDK_ROOT, SNPE_ROOT)
#   - Exports LD_LIBRARY_PATH and PYTHONPATH for SDK + converter venv
#   - Validates that key converter binaries are reachable
#   - Prints a verification summary
#
# The converter venv MUST be Python 3.10 — the SDK's compiled .so files
# link against libpython3.10.so.1.0 and are ABI-incompatible with 3.11+.

set -euo pipefail

# ---- Script directory (repo root) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Defaults --------------------------------------------------------------
# Windows (Git Bash / MSYS) installers put the SDK at C:\Qualcomm\...; Linux and
# macOS installs live under $HOME. Prefer whichever exists on this host, and
# drop a value that is set but points nowhere, so a stale inherited variable
# cannot shadow the default and break the path checks below.
if [[ -n "${QAIRT_SDK_ROOT:-}" && ! -d "$QAIRT_SDK_ROOT" ]]; then
	echo "[qairt-env] Ignoring QAIRT_SDK_ROOT='$QAIRT_SDK_ROOT' (not a directory)." >&2
	unset QAIRT_SDK_ROOT
fi
KAVI_QAIRT_DEFAULT="/c/Qualcomm/AIStack/QAIRT/2.31.0.250130"
if [[ ! -d "$KAVI_QAIRT_DEFAULT" ]]; then
	KAVI_QAIRT_DEFAULT="$HOME/Qualcomm/AIStack/QAIRT/2.31.0.250130"
fi
: "${QAIRT_SDK_ROOT:=$KAVI_QAIRT_DEFAULT}"
unset KAVI_QAIRT_DEFAULT
export QAIRT_SDK_ROOT

# ---- Android NDK (optional) -------------------------------------------------
# Only needed by the qnn-model-lib / context-binary steps (see
# bench/qnn/convert_to_qnn.sh); same Windows-first fallback as above.
if [[ -n "${ANDROID_NDK_ROOT:-}" && ! -d "$ANDROID_NDK_ROOT" ]]; then
	echo "[qairt-env] Ignoring ANDROID_NDK_ROOT='$ANDROID_NDK_ROOT' (not a directory)." >&2
	unset ANDROID_NDK_ROOT
fi
KAVI_NDK_DEFAULT="/c/android_sdk/ndk/26.1.10909125"
if [[ ! -d "$KAVI_NDK_DEFAULT" ]]; then
	KAVI_NDK_DEFAULT="$HOME/Android/Sdk/ndk/26.1.10909125"
fi
if [[ -z "${ANDROID_NDK_ROOT:-}" && -d "$KAVI_NDK_DEFAULT" ]]; then
	export ANDROID_NDK_ROOT="$KAVI_NDK_DEFAULT"
fi
unset KAVI_NDK_DEFAULT

# ---- Resolve real paths ----
QAIRT_SDK_ROOT="$(cd "$QAIRT_SDK_ROOT" 2>/dev/null && pwd)" || {
	echo "[qairt-env] ERROR: QAIRT_SDK_ROOT does not exist or is not a directory: $QAIRT_SDK_ROOT" >&2
	return 1 2>/dev/null || exit 1
}

# ---- Converter venv (Python 3.10, created on first run) ----
VENV_DIR="${REPO_ROOT}/.venv-qairt"
if [[ ! -d "$VENV_DIR" ]]; then
	echo "[qairt-env] Creating converter venv at $VENV_DIR ..."
	if ! command -v uv &>/dev/null; then
		echo "[qairt-env] ERROR: uv is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
		return 1 2>/dev/null || exit 1
	fi
	uv venv --python 3.10 "$VENV_DIR"
	# Use uv pip install (not uv sync) to bypass project's requires-python >=3.12
	uv pip install --python "$VENV_DIR" \
		--index-strategy unsafe-best-match \
		onnx==1.16.1 onnxruntime==1.17.1 "numpy<2" onnx-simplifier \
		scipy lxml absl-py pandas pyyaml
	echo "[qairt-env] Converter venv created."
fi

# Activate the venv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo "[qairt-env] Activated venv: $VENV_DIR (Python $(python3 --version | cut -d' ' -f2))"

# ---- Source SDK envsetup ----
ENVSETUP="$QAIRT_SDK_ROOT/bin/envsetup.sh"
if [[ -f "$ENVSETUP" ]]; then
	# SDK envsetup.sh uses unquoted ${PYTHONPATH} which fails under set -u
	# Temporarily disable nounset for the source call.
	# shellcheck source=/dev/null
	set +u
	source "$ENVSETUP"
	set -u
	echo "[qairt-env] Sourced $ENVSETUP"
else
	echo "[qairt-env] WARNING: $ENVSETUP not found — QNN_SDK_ROOT/SNPE_ROOT may not be set" >&2
fi

# ---- PYTHONPATH (SDK converter packages) ----
SDK_PYTHON="$QAIRT_SDK_ROOT/lib/python"
if [[ -d "$SDK_PYTHON" ]]; then
	export PYTHONPATH="${SDK_PYTHON}${PYTHONPATH:+:$PYTHONPATH}"
fi

# ---- LD_LIBRARY_PATH ----
# Add QAIRT SDK host libs
SDK_LIB="$QAIRT_SDK_ROOT/lib/x86_64-linux-clang"
if [[ -d "$SDK_LIB" ]]; then
	export LD_LIBRARY_PATH="${SDK_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Add the converter venv's Python lib dir (libpython3.10.so.1.0), wherever
# uv installed Python 3.10 on this machine.
UV_PYTHON_LIB="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
if [[ -n "$UV_PYTHON_LIB" && -d "$UV_PYTHON_LIB" ]]; then
	export LD_LIBRARY_PATH="${UV_PYTHON_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ---- Validation ----
BIN_DIR="$QAIRT_SDK_ROOT/bin/x86_64-linux-clang"
echo ""
echo "============================================"
echo " QAIRT Environment — Verification"
echo "============================================"
echo " QAIRT_SDK_ROOT : $QAIRT_SDK_ROOT"
echo " QNN_SDK_ROOT   : ${QNN_SDK_ROOT:-<not set>}"
echo " SNPE_ROOT      : ${SNPE_ROOT:-<not set>}"
echo " ANDROID_NDK_ROOT: ${ANDROID_NDK_ROOT:-<not set>}"
echo " Converter venv : $VENV_DIR"
echo " Python         : $(python3 --version 2>&1)"

# Check key converters
converters=(qnn-onnx-converter qnn-model-lib-generator qnn-context-binary-generator)
all_found=true
for tool in "${converters[@]}"; do
	tool_path="$BIN_DIR/$tool"
	if [[ -x "$tool_path" ]]; then
		echo " $(printf '%25s' "$tool") : $tool_path  ✅"
	else
		echo " $(printf '%25s' "$tool") : NOT FOUND  ❌"
		all_found=false
	fi
done

echo " LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<empty>}"
echo "============================================"

if $all_found; then
	echo "[qairt-env] All key converters are available."
else
	echo "[qairt-env] WARNING: Some converters are missing. Check QAIRT_SDK_ROOT." >&2
fi
