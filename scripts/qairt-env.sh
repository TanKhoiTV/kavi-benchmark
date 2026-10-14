#!/usr/bin/env bash
# shellcheck disable=SC1091
#
# qairt-env.sh — QAIRT environment helper for Linux / WSL (POSIX adapter)
#
# Usage:
#   source scripts/qairt-env.sh
#   QAIRT_SDK_ROOT=/custom/path source scripts/qairt-env.sh
#
# All path and platform resolution lives in scripts/env.py — the single source of
# truth for the SDK root, the converter bin dir, the host library dir, the SDK
# envsetup flavour and the Python 3.10 converter venv. This script only does what
# a shell has to:
#
#   1. refuse to run under Git Bash/MSYS, where uv would build a Scripts/ venv
#      around the Windows converters (use qairt-env.ps1 or WSL2 there)
#   2. create the converter venv if missing and verify the environment
#   3. export the resolved environment and activate the venv
#   4. source the vendor SDK's own envsetup.sh (sets QNN_SDK_ROOT, SNPE_ROOT)
#
# Requires uv: https://docs.astral.sh/uv/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Platform guard ───────────────────────────────────────────────────────────
case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN*)
	echo "[qairt-env] Refusing to run under Git Bash/MSYS: this is the Linux/WSL" >&2
	echo "            helper, and uv here would build a Windows 'Scripts/' venv" >&2
	echo "            around the Windows converters. Use one of:" >&2
	echo "              PowerShell :  . .\\scripts\\qairt-env.ps1" >&2
	echo "              WSL2       :  source scripts/qairt-env.sh" >&2
	return 1 2>/dev/null || exit 1
	;;
esac

# ── Pick an interpreter for scripts/env.py ───────────────────────────────────
# A bare python3 is enough: the resolver is stdlib + PyYAML and stays 3.10
# compatible. It is also the safer choice — `uv run` would sync the project
# environment just to resolve paths, and in a checkout shared with Windows it
# would replace .venv with Linux packages. Escalate only when python3 cannot
# read .kavi.yaml (probing needs the pinned versions).
ENV_PY="${KAVI_PYTHON:-python3}"
if "$ENV_PY" -c 'import yaml' >/dev/null 2>&1; then
	KAVI_ENV=("$ENV_PY" "$REPO_ROOT/scripts/env.py")
else
	echo "[qairt-env] python3 has no PyYAML; falling back to 'uv run' to resolve paths" >&2
	KAVI_ENV=(uv run --project "$REPO_ROOT" python "$REPO_ROOT/scripts/env.py")
fi

# ── Create the venv, then verify before exporting anything ───────────────────
if ! "${KAVI_ENV[@]}" --bootstrap --check; then
	echo "[qairt-env] Environment check failed — fix the problem(s) above and retry." >&2
	return 1 2>/dev/null || exit 1
fi

# ── Export the resolved environment, then activate the converter venv ────────
eval "$("${KAVI_ENV[@]}" --format bash)"
# shellcheck source=/dev/null
source "$KAVI_VENV_ACTIVATE"

# ── Vendor SDK envsetup (a bash-only side effect) ────────────────────────────
if [[ -f "$KAVI_ENVSETUP" ]]; then
	# The SDK's envsetup.sh reads unquoted ${PYTHONPATH}, which fails under set -u.
	set +u
	# shellcheck source=/dev/null
	source "$KAVI_ENVSETUP"
	set -u
else
	echo "[qairt-env] WARNING: $KAVI_ENVSETUP not found — QNN_SDK_ROOT/SNPE_ROOT may be unset" >&2
fi

# ── SDK converter Python packages ────────────────────────────────────────────
if [[ -d "$QAIRT_SDK_ROOT/lib/python" ]]; then
	export PYTHONPATH="$QAIRT_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "[qairt-env] Ready: $(python3 --version 2>&1) from $VIRTUAL_ENV"
