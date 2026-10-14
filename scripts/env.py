"""Resolve Kavi's per-machine toolchain paths — the single source of truth.

The shell adapters (``scripts/qairt-env.sh``, ``scripts/qairt-env.ps1``) and the
QNN conversion pipeline (``bench/qnn/convert_to_qnn.sh``) all ask this module
where the SDK, the NDK, the converter binaries, the host libraries and the
converter venv live, instead of deriving them again themselves. Only four facts
are genuinely platform-dependent; everything else is derived:

    fact              POSIX (WSL / Linux)         Windows (Git Bash / PowerShell)
    venv layout       .venv-qairt/bin             .venv-qairt/Scripts
    converter bin dir bin/x86_64-linux-clang      bin/x86_64-windows-msvc
    host lib dir      lib/x86_64-linux-clang      lib/x86_64-windows-msvc
    SDK envsetup      bin/envsetup.sh             bin/envsetup.ps1

Pinned versions and the environment-variable names come from ``.kavi.yaml``;
per-machine *roots* come from those variables when they point at a real
directory, otherwise from a probe of the usual install locations. A variable
that is set but points nowhere — typically a stale value inherited from Windows
— is ignored with a warning rather than used.

Nothing here needs the network, and ``--bootstrap`` is the only mode that writes
to disk. The module therefore stays importable from the Python 3.10 converter
venv *and* the 3.12 project venv, so either can call it.

Usage::

    eval "$(uv run python -m scripts.env --bootstrap --format bash)"
    Invoke-Expression (uv run python -m scripts.env --format powershell)
    uv run python -m scripts.env --format json
    uv run python -m scripts.env --check
    uv run python -m scripts.env --print converter_bin_dir
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
KAVI_CONFIG_PATH = REPO_ROOT / ".kavi.yaml"

# Names of the per-machine path variables. ``.kavi.yaml`` declares the same
# names in ``sdk.*.path_env`` and wins when it is readable; these defaults let
# the ``--converter-bin-dir`` shortcut work without reading the config at all.
DEFAULT_QAIRT_ENV_VAR = "QAIRT_SDK_ROOT"
DEFAULT_NDK_ENV_VAR = "ANDROID_NDK_ROOT"

# The converter venv is deliberately separate from the project venv: the SDK's
# compiled modules link against libpython3.10, while the project needs >=3.12.
VENV_NAME = ".venv-qairt"
CONVERTER_PYTHON = "3.10"
CONVERTER_TOOLS = (
    "qnn-onnx-converter",
    "qnn-model-lib-generator",
    "qnn-context-binary-generator",
)
# Mirrors [dependency-groups].qairt in pyproject.toml — that file stays the
# source of truth for the list, and tests/test_env.py fails if the two drift.
# The venv is built with `uv pip install` rather than `uv sync` because the
# project's own requires-python (>=3.12) excludes the 3.10 interpreter the SDK
# needs.
CONVERTER_REQUIREMENTS = (
    "onnx==1.16.1",
    "onnxruntime==1.17.1",
    "numpy<2",
    "onnx-simplifier",
    "scipy",
    "lxml",
    "absl-py",
    "pandas",
    "pyyaml",
)

_PRINTABLE = (
    "platform",
    "qairt_version",
    "ndk_version",
    "qairt_root",
    "ndk_root",
    "converter_bin_dir",
    "host_lib_dir",
    "sdk_python_dir",
    "envsetup_path",
    "venv_dir",
    "venv_python",
    "venv_activate",
)


# ── platform layout ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Platform:
    """The platform-dependent facts — the only OS-coupled part of the setup."""

    name: str  # "windows" | "posix"
    triple: str  # host toolchain directory under the SDK's bin/ and lib/
    venv_bin: str  # venv subdirectory holding the interpreter
    venv_python: str
    venv_activate: str  # venv activation script the shell should source
    envsetup: str
    lib_path_var: str  # how the loader finds host libraries
    qairt_prefixes: tuple[Path, ...]  # install roots to probe, in order
    ndk_prefixes: tuple[Path, ...]


def detect_platform() -> str:
    """Return ``"windows"`` on Windows (including Git Bash/MSYS), else ``"posix"``.

    Git Bash runs a *Windows* interpreter, so it is correctly reported as
    ``"windows"``: the converters and the venv layout are the Windows ones.
    """
    return "windows" if os.name == "nt" else "posix"


def make_platform(name: str, *, home: Path, environ: Mapping[str, str]) -> Platform:
    """Build the layout for *name*, with candidate install roots in probe order."""
    if name == "windows":
        # Windows installers use C:\Qualcomm and C:\android_sdk; some setups put
        # both under the user profile instead. Git Bash sees the same roots, so
        # --format bash prints them as C:/… for the very same directories.
        return Platform(
            name=name,
            triple="x86_64-windows-msvc",
            venv_bin="Scripts",
            venv_python="python.exe",
            venv_activate="Activate.ps1",
            envsetup="envsetup.ps1",
            lib_path_var="PATH",
            qairt_prefixes=(Path("C:/Qualcomm"), home / "Qualcomm"),
            ndk_prefixes=_ndk_prefixes(
                (Path("C:/android_sdk/ndk"), home / "Android/Sdk/ndk"), environ
            ),
        )
    # POSIX: a native install first, then the Windows drives as WSL sees them.
    return Platform(
        name=name,
        triple="x86_64-linux-clang",
        venv_bin="bin",
        venv_python="python",
        venv_activate="activate",
        envsetup="envsetup.sh",
        lib_path_var="LD_LIBRARY_PATH",
        qairt_prefixes=(
            home / "Qualcomm",
            Path("/mnt/c/Qualcomm"),
            Path("/c/Qualcomm"),
        ),
        ndk_prefixes=_ndk_prefixes(
            (
                home / "Android/Sdk/ndk",
                Path("/mnt/c/android_sdk/ndk"),
                Path("/c/android_sdk/ndk"),
            ),
            environ,
        ),
    )


def _ndk_prefixes(
    defaults: Sequence[Path], environ: Mapping[str, str]
) -> tuple[Path, ...]:
    """Prefer the NDK under ``$ANDROID_SDK_ROOT``/``$ANDROID_HOME``, then *defaults*."""
    prefixes: list[Path] = []
    for var in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        value = environ.get(var, "").strip()
        if value:
            prefixes.append(Path(value) / "ndk")
    prefixes.extend(defaults)
    return tuple(prefixes)


# ── pinned config ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KaviConfig:
    """The pinned values this module needs from ``.kavi.yaml``."""

    qairt_version: str
    ndk_version: str
    qairt_env_var: str
    ndk_env_var: str


def load_kavi_config(path: Path | None = None) -> KaviConfig:
    """Read the SDK/NDK pins and path-variable names from ``.kavi.yaml``."""
    try:
        import yaml
    except ModuleNotFoundError as exc:  # bare interpreter without the qairt group
        raise RuntimeError(
            "PyYAML is required to read .kavi.yaml — run this module through "
            "`uv run python -m scripts.env`, or use the .venv-qairt venv"
        ) from exc

    config_path = path or KAVI_CONFIG_PATH
    try:
        with open(config_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Kavi config not found: {config_path}") from exc

    try:
        qairt = data["sdk"]["qairt"]
        ndk = data["sdk"]["ndk"]
        return KaviConfig(
            qairt_version=str(qairt["version"]),
            ndk_version=str(ndk["version"]),
            qairt_env_var=str(qairt["path_env"]),
            ndk_env_var=str(ndk["path_env"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{config_path}: expected sdk.qairt / sdk.ndk entries ({exc})"
        ) from exc


# ── resolution ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolvedEnv:
    """Everything the shells and the conversion pipeline need, resolved once."""

    platform: Platform
    repo_root: Path
    qairt_version: str
    ndk_version: str
    qairt_env_var: str
    ndk_env_var: str
    qairt_root: Path
    ndk_root: Path | None
    qairt_source: str  # argument | environment | probe | none
    ndk_source: str

    # ── derived paths ──

    @property
    def converter_bin_dir(self) -> Path:
        return self.qairt_root / "bin" / self.platform.triple

    @property
    def host_lib_dir(self) -> Path:
        return self.qairt_root / "lib" / self.platform.triple

    @property
    def sdk_python_dir(self) -> Path:
        return self.qairt_root / "lib" / "python"

    @property
    def envsetup_path(self) -> Path:
        return self.qairt_root / "bin" / self.platform.envsetup

    @property
    def venv_dir(self) -> Path:
        return self.repo_root / VENV_NAME

    @property
    def venv_bin_dir(self) -> Path:
        return self.venv_dir / self.platform.venv_bin

    @property
    def venv_python(self) -> Path:
        return self.venv_bin_dir / self.platform.venv_python

    @property
    def venv_activate(self) -> Path:
        return self.venv_bin_dir / self.platform.venv_activate

    # ── lookups ──

    def find_tool(self, tool: str) -> Path | None:
        """Locate *tool* in the converter bin dir.

        Windows ships a mix of suffixes for the same tool (``.exe``, ``.bat``, or
        no extension at all when it is a Python script), so match on the stem.
        """
        return _find_tool(self.converter_bin_dir, tool)

    def venv_python_version(self) -> str | None:
        """Version reported by the converter venv, or ``None`` if it is absent."""
        if not self.venv_python.is_file():
            return None
        out = _run_python(
            self.venv_python, "import platform; print(platform.python_version())"
        )
        return out or None

    def python_libdir(self) -> Path | None:
        """``libpython3.x`` directory of the converter venv, for ``LD_LIBRARY_PATH``.

        Only meaningful on POSIX; Windows finds the DLLs on ``PATH``.
        """
        if self.platform.name == "windows":
            return None
        code = "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')"
        out = _run_python(self.venv_python, code)
        return Path(out) if out else None

    # ── checks ──

    def problems(self) -> list[str]:
        """Human-readable reasons the environment is unusable (empty if fine)."""
        if not self.qairt_root.is_dir():
            probed = ", ".join(p.as_posix() for p in self.platform.qairt_prefixes)
            return [
                f"{self.qairt_env_var} is not set to an existing directory "
                f"(resolved: {self.qairt_root.as_posix()}); probed: {probed}"
            ]

        problems: list[str] = []
        if not self.converter_bin_dir.is_dir():
            problems.append(
                f"converter bin dir missing: {self.converter_bin_dir.as_posix()}"
            )
        else:
            for tool in CONVERTER_TOOLS:
                path = self.find_tool(tool)
                if path is None:
                    problems.append(
                        f"converter not found: {tool} in {self.converter_bin_dir.as_posix()}"
                    )
                elif self.platform.name == "posix" and not _is_executable(path):
                    problems.append(f"converter not executable: {path.as_posix()}")
        if not self.envsetup_path.is_file():
            problems.append(f"SDK envsetup missing: {self.envsetup_path.as_posix()}")
        if self.ndk_root is None:
            problems.append(f"{self.ndk_env_var} is not set to an existing directory")
        elif not self.ndk_root.is_dir():
            problems.append(
                f"{self.ndk_env_var} does not exist: {self.ndk_root.as_posix()}"
            )

        if not self.venv_python.is_file():
            problems.append(
                f"converter venv missing: {self.venv_python.as_posix()} (run with --bootstrap)"
            )
        else:
            version = self.venv_python_version()
            if version and not version.startswith(CONVERTER_PYTHON):
                problems.append(
                    f"converter venv is Python {version}, but the QAIRT SDK needs {CONVERTER_PYTHON}"
                )
        return problems

    # ── output ──

    def to_env(self, *, style: str = "posix") -> dict[str, str]:
        """The variables the shell adapters export, keyed by variable name.

        *style* selects the path spelling: ``posix`` (``/home/x``, ``C:/x``),
        ``native`` (``C:\\x``) or ``msys`` (``/c/x``, what Git Bash expects).
        """
        env = {
            self.qairt_env_var: _shell_path(self.qairt_root, style),
            "VIRTUAL_ENV": _shell_path(self.venv_dir, style),
            "KAVI_CONVERTER_BIN_DIR": _shell_path(self.converter_bin_dir, style),
            "KAVI_HOST_LIB_DIR": _shell_path(self.host_lib_dir, style),
            "KAVI_ENVSETUP": _shell_path(self.envsetup_path, style),
            "KAVI_VENV_DIR": _shell_path(self.venv_dir, style),
            "KAVI_VENV_PYTHON": _shell_path(self.venv_python, style),
            "KAVI_VENV_ACTIVATE": _shell_path(self.venv_activate, style),
        }
        if self.ndk_root is not None:
            env[self.ndk_env_var] = _shell_path(self.ndk_root, style)
        return env

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable view (POSIX-style paths, so it is host-independent)."""
        return {
            "platform": self.platform.name,
            "qairt_version": self.qairt_version,
            "ndk_version": self.ndk_version,
            "qairt_root": self.qairt_root.as_posix(),
            "qairt_source": self.qairt_source,
            "ndk_root": self.ndk_root.as_posix() if self.ndk_root else None,
            "ndk_source": self.ndk_source,
            "converter_bin_dir": self.converter_bin_dir.as_posix(),
            "host_lib_dir": self.host_lib_dir.as_posix(),
            "sdk_python_dir": self.sdk_python_dir.as_posix(),
            "envsetup_path": self.envsetup_path.as_posix(),
            "venv_dir": self.venv_dir.as_posix(),
            "venv_python": self.venv_python.as_posix(),
            "converters": {
                tool: (path.as_posix() if (path := self.find_tool(tool)) else None)
                for tool in CONVERTER_TOOLS
            },
            "problems": self.problems(),
        }


def resolve(
    config: KaviConfig,
    *,
    platform_name: str | None = None,
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
    qairt_root: Path | None = None,
    ndk_root: Path | None = None,
) -> ResolvedEnv:
    """Resolve every path, preferring an explicit argument, then the environment
    variable, then a probe of the usual install locations."""
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else home
    platform = make_platform(
        platform_name or detect_platform(), home=home, environ=environ
    )

    qairt_candidates = [
        prefix / "AIStack" / "QAIRT" / config.qairt_version
        for prefix in platform.qairt_prefixes
    ]
    qairt, qairt_source = _pick_root(
        var=config.qairt_env_var,
        explicit=qairt_root,
        environ=environ,
        candidates=qairt_candidates,
    )
    ndk, ndk_source = _pick_root(
        var=config.ndk_env_var,
        explicit=ndk_root,
        environ=environ,
        candidates=[prefix / config.ndk_version for prefix in platform.ndk_prefixes],
    )

    return ResolvedEnv(
        platform=platform,
        repo_root=(Path(repo_root) if repo_root is not None else REPO_ROOT),
        qairt_version=config.qairt_version,
        ndk_version=config.ndk_version,
        qairt_env_var=config.qairt_env_var,
        ndk_env_var=config.ndk_env_var,
        # Fall back to the first candidate so diagnostics name the path we wanted.
        qairt_root=qairt if qairt is not None else qairt_candidates[0],
        ndk_root=ndk,
        qairt_source=qairt_source,
        ndk_source=ndk_source,
    )


def converter_bin_dir_for(
    *,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    qairt_root: Path | None = None,
    home: Path | None = None,
) -> Path:
    """Return the converter bin dir *without* reading ``.kavi.yaml``.

    The pinned version only matters when *probing* install locations; once the
    SDK root is known the bin dir is just ``bin/<platform triple>``. The
    conversion pipeline needs exactly this and must keep working under a bare
    ``python3`` that has no PyYAML.

    Raises ``ValueError`` when the SDK root is missing or not a directory.
    """
    environ = os.environ if environ is None else environ
    platform = make_platform(
        platform_name or detect_platform(),
        home=Path.home() if home is None else home,
        environ=environ,
    )
    value = (
        str(qairt_root)
        if qairt_root is not None
        else environ.get(DEFAULT_QAIRT_ENV_VAR, "")
    )
    if not value.strip() or not Path(value).is_dir():
        raise ValueError(
            f"{DEFAULT_QAIRT_ENV_VAR} is not set to an existing directory: {value!r}"
        )
    return Path(value) / "bin" / platform.triple


def _pick_root(
    *,
    var: str,
    explicit: Path | None,
    environ: Mapping[str, str],
    candidates: Sequence[Path],
) -> tuple[Path | None, str]:
    """Return ``(root, source)`` for one SDK root, ignoring unusable values."""
    if explicit is not None:
        return Path(explicit), "argument"

    value = environ.get(var, "").strip()
    if value:
        if Path(value).is_dir():
            return Path(value), "environment"
        print(
            f"[scripts.env] ignoring {var}='{value}' (not a directory); probing defaults",
            file=sys.stderr,
        )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate, "probe"
    return None, "none"


def _find_tool(bin_dir: Path, tool: str) -> Path | None:
    """Find *tool* in *bin_dir*, tolerating Windows' varying suffixes."""
    if not bin_dir.is_dir():
        return None
    for candidate in sorted(bin_dir.glob(f"{tool}*")):
        if candidate.is_file():
            return candidate
    return None


def _is_executable(path: Path) -> bool:
    """True when *path* carries the executable bit (meaningful on POSIX only)."""
    try:
        return os.access(path, os.X_OK)
    except OSError:
        return False


def _run_python(python: Path, code: str) -> str:
    """Run a one-liner with *python*, returning stripped stdout (``""`` on error)."""
    try:
        result = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            # The console code page is not UTF-8 on Windows (cp932 here); never
            # let a decode error turn into a traceback in the middle of a check.
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


# ── shell output ────────────────────────────────────────────────────────────


def _shell_path(path: Path, style: str) -> str:
    """Render *path* for *style*: ``posix`` (as-is), ``native`` (Windows), ``msys``.

    Git Bash is a Windows host with a POSIX shell: ``C:/Qualcomm/x`` inside a
    ``:``-separated ``PATH`` is ambiguous with the drive letter, so emit the
    ``/c/Qualcomm/x`` spelling there instead.
    """
    if style == "native":
        return str(path)
    posix = path.as_posix()
    if style == "msys" and len(posix) > 2 and posix[1] == ":" and posix[2] == "/":
        return f"/{posix[0].lower()}{posix[2:]}"
    return posix


def _prepend(
    var: str, entries: Sequence[Path], *, separator: str, style: str = "posix"
) -> str:
    """``export VAR='a:b'"${VAR:-}"`` — quoted literal plus a live expansion."""
    if not entries:
        return f"# {var}: nothing to prepend"
    prefix = separator.join(_shell_path(entry, style) for entry in entries)
    return f'export {var}={shlex.quote(prefix)}"${{{var}:-}}"'


def _path_prepends(env: ResolvedEnv) -> list[tuple[str, list[Path]]]:
    """(variable, directories) pairs, per platform.

    Windows resolves host DLLs through ``PATH`` like anything else; POSIX uses
    ``LD_LIBRARY_PATH``, which also has to carry the venv's ``libpython3.10``.
    The list separator belongs to the *shell*, not the platform — Git Bash keeps
    ``$PATH`` POSIX-separated even though it is a Windows host — so each
    formatter joins with its own separator.
    """
    path_entries = [env.converter_bin_dir, env.venv_bin_dir]
    if env.platform.lib_path_var == "PATH":
        return [("PATH", [*path_entries, env.host_lib_dir])]
    lib_entries = [env.host_lib_dir]
    libdir = env.python_libdir()
    if libdir is not None:
        lib_entries.append(libdir)
    return [("PATH", path_entries), (env.platform.lib_path_var, lib_entries)]


def _separator(var: str, shell: str) -> str:
    """List separator for *var* in *shell* (only ``PATH`` is ``;``-joined)."""
    if shell == "powershell" and var == "PATH":
        return ";"
    return ":"


def format_bash(env: ResolvedEnv) -> str:
    """``export`` statements plus PATH/LD_LIBRARY_PATH prepends, ready to ``eval``.

    Targets Git Bash/MSYS on Windows and POSIX shells elsewhere, so Windows
    hosts get the ``/c/...`` spelling Git Bash uses natively.
    """
    style = "msys" if env.platform.name == "windows" else "posix"
    lines = ["# Generated by scripts/env.py - do not edit."]
    lines += [
        f"export {key}={shlex.quote(value)}"
        for key, value in env.to_env(style=style).items()
    ]
    lines += [
        _prepend(var, entries, separator=_separator(var, "bash"), style=style)
        for var, entries in _path_prepends(env)
    ]
    return "\n".join(lines) + "\n"


def _ps_quote(value: str) -> str:
    """Single-quote *value* for PowerShell (an embedded quote is doubled)."""
    return "'" + value.replace("'", "''") + "'"


def format_powershell(env: ResolvedEnv) -> str:
    """PowerShell assignments plus PATH prepends, ready for ``Invoke-Expression``."""
    lines = ["# Generated by scripts/env.py - do not edit."]
    lines += [
        f"$env:{key} = {_ps_quote(value)}"
        for key, value in env.to_env(style="native").items()
    ]
    for var, entries in _path_prepends(env):
        separator = _separator(var, "powershell")
        prefix = separator.join(str(entry) for entry in entries) + separator
        lines.append(f"$env:{var} = {_ps_quote(prefix)} + $env:{var}")
    if env.sdk_python_dir.is_dir():
        pythonpath = str(env.sdk_python_dir) + ";"
        lines.append(f"$env:PYTHONPATH = {_ps_quote(pythonpath)} + $env:PYTHONPATH")
    return "\n".join(lines) + "\n"


def format_json(env: ResolvedEnv) -> str:
    return json.dumps(env.as_dict(), indent=2) + "\n"


FORMATTERS = {"bash": format_bash, "powershell": format_powershell, "json": format_json}


# ── actions ─────────────────────────────────────────────────────────────────


def bootstrap(env: ResolvedEnv) -> int:
    """Create the Python 3.10 converter venv if it is missing; return exit status."""
    if env.venv_python.is_file():
        print(
            f"[scripts.env] converter venv already present: {env.venv_dir}",
            file=sys.stderr,
        )
        return 0

    uv = shutil.which("uv")
    if uv is None:
        print(
            "[scripts.env] uv is not installed (https://docs.astral.sh/uv/) — needed to "
            f"create the Python {CONVERTER_PYTHON} converter venv",
            file=sys.stderr,
        )
        return 1

    steps = [
        [uv, "venv", "--python", CONVERTER_PYTHON, str(env.venv_dir)],
        [
            uv,
            "pip",
            "install",
            "--python",
            str(env.venv_dir),
            # The SDK pins old onnx/numpy versions that the project's own
            # resolution would otherwise upgrade past.
            "--index-strategy",
            "unsafe-best-match",
            *CONVERTER_REQUIREMENTS,
        ],
    ]
    print(
        f"[scripts.env] creating converter venv (Python {CONVERTER_PYTHON}) at {env.venv_dir}",
        file=sys.stderr,
    )
    for step in steps:
        print(f"[scripts.env] $ {' '.join(step)}", file=sys.stderr)
        try:
            # Inherit stdio so uv's own progress is visible while it runs.
            subprocess.run(step, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"[scripts.env] failed: {exc}", file=sys.stderr)
            return 1
    return 0


def report(env: ResolvedEnv) -> int:
    """Print a verification summary; return 1 when something is wrong."""
    problems = env.problems()
    rows = [
        ("platform", f"{env.platform.name} (python {sys.version.split()[0]})"),
        (env.qairt_env_var, f"{env.qairt_root}  [{env.qairt_source}]"),
        ("converter bin dir", env.converter_bin_dir.as_posix()),
        ("SDK envsetup", env.envsetup_path.as_posix()),
        (
            env.ndk_env_var,
            f"{env.ndk_root}  [{env.ndk_source}]"
            if env.ndk_root
            else f"<not found>  [{env.ndk_source}]",
        ),
        (
            "converter venv",
            f"{env.venv_dir}  (python {env.venv_python_version() or 'missing'})",
        ),
    ]
    width = max(len(label) for label, _ in rows)
    line = "=" * (width + 2 + 60)
    print(line)
    print(" QAIRT Environment - Verification")
    print(line)
    for label, value in rows:
        print(f" {label:<{width}} : {value}")
    print(f" {'converters':<{width}} :")
    for tool in CONVERTER_TOOLS:
        path = env.find_tool(tool)
        state = "OK" if path is not None else "NOT FOUND"
        print(f"   {tool:<40} {state:<10} {path or ''}")
    print("")
    if problems:
        print(f" {len(problems)} problem(s):")
        for problem in problems:
            print(f"   - {problem}")
        return 1
    print(" Environment OK.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.env",
        description="Resolve Kavi's SDK/NDK/converter paths (single source of truth).",
    )
    parser.add_argument(
        "--format",
        choices=sorted(FORMATTERS),
        default=None,
        help="Emit the environment for this shell. Without it the environment is verified.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help=f"Create {VENV_NAME} (Python {CONVERTER_PYTHON}) if it is missing.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the resolved environment and exit non-zero on problems.",
    )
    parser.add_argument(
        "--print",
        dest="print_key",
        metavar="KEY",
        choices=_PRINTABLE,
        help=f"Print one resolved value. Keys: {', '.join(_PRINTABLE)}",
    )
    parser.add_argument(
        "--converter-bin-dir",
        action="store_true",
        help="Print the platform's converter bin dir (reads no .kavi.yaml).",
    )
    parser.add_argument(
        "--qairt-root", type=Path, default=None, help="Override the QAIRT SDK root."
    )
    parser.add_argument(
        "--ndk-root", type=Path, default=None, help="Override the Android NDK root."
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "windows", "posix"),
        default="auto",
        help="Assume this platform layout instead of detecting it.",
    )
    parser.add_argument(
        "--home", type=Path, default=None, help="Home directory used for probing."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: the real one).",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to .kavi.yaml.")
    args = parser.parse_args(argv)
    platform_name = None if args.platform == "auto" else args.platform

    # Shortcut for bench/qnn/convert_to_qnn.sh: derived from the SDK root and the
    # platform alone, so it never needs PyYAML or the pinned versions.
    if args.converter_bin_dir:
        try:
            print(
                converter_bin_dir_for(
                    platform_name=platform_name,
                    qairt_root=args.qairt_root,
                    home=args.home,
                ).as_posix()
            )
        except ValueError as exc:
            print(f"[scripts.env] {exc}", file=sys.stderr)
            return 1
        return 0

    env = resolve(
        load_kavi_config(args.config),
        platform_name=platform_name,
        home=args.home,
        repo_root=args.repo_root,
        qairt_root=args.qairt_root,
        ndk_root=args.ndk_root,
    )

    status = bootstrap(env) if args.bootstrap else 0
    if args.check:
        status |= report(env)
    elif args.print_key:
        value: Any = (
            env.platform.name
            if args.print_key == "platform"
            else getattr(env, args.print_key)
        )
        print(
            ""
            if value is None
            else (value.as_posix() if isinstance(value, Path) else value)
        )
    elif args.format:
        print(FORMATTERS[args.format](env), end="")
    else:
        status |= report(env)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
