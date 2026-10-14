"""Tests for ``scripts.env`` — the single source of truth for toolchain paths.

Every test builds fake SDK/NDK trees under ``tmp_path``: no real QAIRT SDK, no
network, and both platform layouts are exercised on any host, so this suite runs
unchanged on the Linux CI runner and on a Windows dev box.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from scripts.env import (
    CONVERTER_REQUIREMENTS,
    CONVERTER_TOOLS,
    KAVI_CONFIG_PATH,
    REPO_ROOT,
    KaviConfig,
    _ps_quote,
    _shell_path,
    detect_platform,
    format_bash,
    format_json,
    format_powershell,
    load_kavi_config,
    make_platform,
    report,
    resolve,
)

QAIRT_VERSION = "2.31.0.250130"
NDK_VERSION = "26.1.10909125"
WINDOWS_TRIPLE = "x86_64-windows-msvc"
POSIX_TRIPLE = "x86_64-linux-clang"

CONFIG = KaviConfig(
    qairt_version=QAIRT_VERSION,
    ndk_version=NDK_VERSION,
    qairt_env_var="QAIRT_SDK_ROOT",
    ndk_env_var="ANDROID_NDK_ROOT",
)


def fake_sdk(
    home: Path,
    *,
    triple: str,
    tools: tuple[str, ...] = CONVERTER_TOOLS,
    envsetup: str = "envsetup.sh",
) -> Path:
    """Create a fake QAIRT SDK under *home* and return its root."""
    root = home / "Qualcomm" / "AIStack" / "QAIRT" / QAIRT_VERSION
    bin_dir = root / "bin" / triple
    bin_dir.mkdir(parents=True)
    for tool in tools:
        path = bin_dir / tool
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
    (root / "lib" / triple).mkdir(parents=True)
    (root / "lib" / "python").mkdir(parents=True)
    (root / "bin" / envsetup).write_text("", encoding="utf-8")
    return root


def fake_ndk(home: Path) -> Path:
    """Create a fake NDK under *home*'s Android SDK and return its root."""
    root = home / "Android" / "Sdk" / "ndk" / NDK_VERSION
    root.mkdir(parents=True)
    return root


def fake_venv(repo_root: Path, platform_name: str) -> Path:
    """Create a placeholder converter venv (a stub interpreter, not a real one)."""
    layout = make_platform(platform_name, home=Path("/"), environ={})
    interpreter = repo_root / ".venv-qairt" / layout.venv_bin / layout.venv_python
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    return interpreter


def resolved(
    tmp_path: Path, platform_name: str, *, environ: dict[str, str] | None = None
):
    triple = WINDOWS_TRIPLE if platform_name == "windows" else POSIX_TRIPLE
    envsetup = "envsetup.ps1" if platform_name == "windows" else "envsetup.sh"
    fake_sdk(tmp_path, triple=triple, envsetup=envsetup)
    fake_ndk(tmp_path)
    repo_root = tmp_path / "repo"
    fake_venv(repo_root, platform_name)
    return resolve(
        CONFIG,
        platform_name=platform_name,
        home=tmp_path,
        environ=environ or {},
        repo_root=repo_root,
    )


# ── platform layouts ────────────────────────────────────────────────────────


class TestLayouts:
    """The four platform-dependent facts, one test per layout."""

    def test_windows_uses_msvc_triple_and_scripts_venv(self, tmp_path: Path) -> None:
        env = resolved(tmp_path, "windows")
        assert env.platform.triple == WINDOWS_TRIPLE
        assert env.converter_bin_dir.name == WINDOWS_TRIPLE
        assert env.envsetup_path.name == "envsetup.ps1"
        assert env.venv_python.parent.name == "Scripts"
        assert env.venv_python.name == "python.exe"

    def test_posix_uses_linux_clang_triple_and_bin_venv(self, tmp_path: Path) -> None:
        env = resolved(tmp_path, "posix")
        assert env.platform.triple == POSIX_TRIPLE
        assert env.converter_bin_dir.name == POSIX_TRIPLE
        assert env.envsetup_path.name == "envsetup.sh"
        assert env.venv_python.parent.name == "bin"
        assert env.venv_python.name == "python"

    def test_windows_tool_lookup_tolerates_mixed_suffixes(self, tmp_path: Path) -> None:
        """Windows ships the same three tools with .exe, .bat *and* no extension."""
        fake_sdk(
            tmp_path,
            triple=WINDOWS_TRIPLE,
            tools=(
                "qnn-onnx-converter",
                "qnn-model-lib-generator.bat",
                "qnn-context-binary-generator.exe",
            ),
            envsetup="envsetup.ps1",
        )
        fake_ndk(tmp_path)
        env = resolve(CONFIG, platform_name="windows", home=tmp_path, environ={})
        for tool in CONVERTER_TOOLS:
            assert env.find_tool(tool) is not None, tool

    def test_detection_matches_the_host(self) -> None:
        assert detect_platform() == ("windows" if os.name == "nt" else "posix")


# ── root resolution ─────────────────────────────────────────────────────────


class TestRootResolution:
    """Explicit argument > environment variable > probe; unusable values dropped."""

    def test_probe_finds_the_usual_layout(self, tmp_path: Path) -> None:
        env = resolved(tmp_path, "posix")
        assert env.qairt_source == "probe"
        assert env.ndk_source == "probe"
        assert env.qairt_root.is_dir()
        assert env.ndk_root is not None and env.ndk_root.is_dir()

    def test_environment_variable_wins(self, tmp_path: Path) -> None:
        sdk = fake_sdk(tmp_path / "elsewhere", triple=POSIX_TRIPLE)
        ndk = fake_ndk(tmp_path / "elsewhere")
        env = resolve(
            CONFIG,
            platform_name="posix",
            home=tmp_path,
            environ={"QAIRT_SDK_ROOT": str(sdk), "ANDROID_NDK_ROOT": str(ndk)},
        )
        assert (env.qairt_source, env.ndk_source) == ("environment", "environment")
        assert env.qairt_root == sdk
        assert env.ndk_root == ndk

    def test_stale_environment_variable_is_ignored(
        self, tmp_path: Path, capsys
    ) -> None:
        """Regression: a variable set to a non-directory must not shadow the probe.

        This is the real failure seen on Windows, where ANDROID_NDK_ROOT had been
        mangled to a literal ``\\ndk\\...`` and silently beat the default.
        """
        env = resolved(
            tmp_path, "posix", environ={"ANDROID_NDK_ROOT": r"\ndk\26.1.10909125"}
        )
        assert env.ndk_source == "probe"
        assert env.ndk_root is not None and env.ndk_root.is_dir()
        assert "ignoring ANDROID_NDK_ROOT" in capsys.readouterr().err

    def test_explicit_argument_wins_over_environment(self, tmp_path: Path) -> None:
        explicit = fake_sdk(tmp_path / "explicit", triple=POSIX_TRIPLE)
        other = fake_sdk(tmp_path / "other", triple=POSIX_TRIPLE)
        env = resolve(
            CONFIG,
            platform_name="posix",
            home=tmp_path / "home",
            environ={"QAIRT_SDK_ROOT": str(other)},
            qairt_root=explicit,
            repo_root=tmp_path / "repo",
        )
        assert (env.qairt_root, env.qairt_source) == (explicit, "argument")

    def test_android_sdk_root_env_is_probed(self, tmp_path: Path) -> None:
        sdk_root = tmp_path / "andersdk"
        ndk = sdk_root / "ndk" / NDK_VERSION
        ndk.mkdir(parents=True)
        env = resolve(
            CONFIG,
            platform_name="posix",
            home=tmp_path / "empty-home",
            environ={"ANDROID_SDK_ROOT": str(sdk_root)},
        )
        assert env.ndk_root == ndk

    def test_missing_sdk_names_the_environment_variable(self, tmp_path: Path) -> None:
        """Nothing installed anywhere: the message must name the variable to set.

        The version is deliberately unmatchable so the probe cannot hit a real
        SDK at one of the candidate locations on the *host* running the suite
        (inside WSL, /mnt/c/Qualcomm is a live install).
        """
        impossible = KaviConfig(
            qairt_version="0.0.0-none",
            ndk_version="0.0.0-none",
            qairt_env_var="QAIRT_SDK_ROOT",
            ndk_env_var="ANDROID_NDK_ROOT",
        )
        env = resolve(
            impossible,
            platform_name="posix",
            home=tmp_path,
            environ={},
            repo_root=tmp_path / "repo",
        )
        assert env.qairt_source == "none"
        problems = env.problems()
        assert problems
        assert "QAIRT_SDK_ROOT" in problems[0]


# ── checks ──────────────────────────────────────────────────────────────────


class TestChecks:
    def test_complete_layout_has_no_problems(self, tmp_path: Path) -> None:
        env = resolved(tmp_path, "posix")
        assert env.problems() == []

    def test_missing_venv_is_reported(self, tmp_path: Path) -> None:
        env = resolved(tmp_path, "posix")
        env.venv_python.unlink()
        assert any("--bootstrap" in problem for problem in env.problems())

    def test_non_executable_converter_is_reported(self, tmp_path: Path) -> None:
        if os.name == "nt":
            pytest.skip("POSIX mode bits do not exist on Windows")
        env = resolved(tmp_path, "posix")
        tool = env.find_tool("qnn-onnx-converter")
        assert tool is not None
        tool.chmod(0o644)
        assert any("not executable" in problem for problem in env.problems())

    def test_report_exit_status(self, tmp_path: Path, capsys) -> None:
        env = resolved(tmp_path, "posix")
        assert report(env) == 0
        env.venv_python.unlink()
        assert report(env) == 1
        capsys.readouterr()


# ── shell output ────────────────────────────────────────────────────────────


class TestOutput:
    def test_output_is_pure_ascii(self, tmp_path: Path) -> None:
        """Regression: the Windows console code page (cp932 here) rejects em dashes."""
        env = resolved(tmp_path, "windows")
        for payload in (format_bash(env), format_powershell(env), format_json(env)):
            payload.encode("ascii")

    def test_msys_path_spelling(self) -> None:
        """Drive-letter paths become /c/... for Git Bash; everything else is left alone.

        Uses literal Windows-style paths rather than ``tmp_path``: on the Linux CI
        runner ``tmp_path`` has no drive letter, so asserting on it would only
        hold on a Windows host.
        """
        assert _shell_path(Path("C:/Qualcomm/AIStack"), "msys") == "/c/Qualcomm/AIStack"
        assert _shell_path(Path("D:/android_sdk/ndk"), "msys") == "/d/android_sdk/ndk"
        # Already-POSIX paths, and the posix/native spellings, are untouched.
        assert _shell_path(Path("/home/dmin/Qualcomm"), "msys") == "/home/dmin/Qualcomm"
        assert (
            _shell_path(Path("C:/Qualcomm/AIStack"), "posix") == "C:/Qualcomm/AIStack"
        )
        assert _shell_path(Path("C:/Qualcomm/AIStack"), "native") == str(
            Path("C:/Qualcomm/AIStack")
        )

    def test_format_bash_rewrites_windows_roots_for_git_bash(
        self, tmp_path: Path
    ) -> None:
        """The whole payload, not just the helper: a Windows root reaches bash as /c/..."""
        env = resolve(
            CONFIG,
            platform_name="windows",
            home=tmp_path,
            environ={},
            qairt_root=Path("C:/fake/Qualcomm/AIStack/QAIRT/2.31.0.250130"),
        )
        payload = format_bash(env)
        assert (
            "export QAIRT_SDK_ROOT=/c/fake/Qualcomm/AIStack/QAIRT/2.31.0.250130"
            in payload
        )
        assert "C:/fake" not in payload

    def test_bash_output_is_sourceable(self, tmp_path: Path) -> None:
        """Whatever the layout, the emitted payload must set the variables.

        Fed through stdin rather than a file path: Git Bash mangles a path passed
        in argv, so this keeps the test about the payload, not about MSYS.
        """
        if shutil.which("bash") is None:
            pytest.skip("bash not available")
        env = resolved(tmp_path, "posix")
        script = format_bash(env) + 'printf %s "$KAVI_CONVERTER_BIN_DIR"'
        result = subprocess.run(
            ["bash", "-s"],
            # Binary both ways: text-mode stdin on Windows rewrites \n to \r\n,
            # and a stray CR inside an unquoted export value is exactly the
            # corruption the .kavi.env* eol=lf rule in .gitattributes prevents.
            input=script.encode("utf-8"),
            capture_output=True,
            check=True,
        )
        assert result.stdout.decode("utf-8") == env.converter_bin_dir.as_posix()

    def test_bash_output_prepends_both_paths(self, tmp_path: Path) -> None:
        env = resolved(tmp_path, "posix")
        lines = format_bash(env).splitlines()
        path_line = next(line for line in lines if line.startswith("export PATH="))
        assert env.converter_bin_dir.as_posix() in path_line
        assert env.venv_bin_dir.as_posix() in path_line
        assert "${PATH:-}" in path_line
        lib_line = next(
            line for line in lines if line.startswith("export LD_LIBRARY_PATH=")
        )
        assert env.host_lib_dir.as_posix() in lib_line

    def test_windows_path_variable_is_path_not_ld_library_path(
        self, tmp_path: Path
    ) -> None:
        env = resolved(tmp_path, "windows")
        assert "LD_LIBRARY_PATH" not in format_bash(env)

    def test_powershell_escapes_embedded_quotes(self) -> None:
        assert _ps_quote("C:\\it's here") == "'C:\\it''s here'"

    def test_powershell_output_assigns_each_variable(self, tmp_path: Path) -> None:
        env = resolved(tmp_path, "windows")
        payload = format_powershell(env)
        assert f"$env:QAIRT_SDK_ROOT = '{env.qairt_root}" in payload
        assert "+ $env:PATH" in payload

    def test_json_round_trips(self, tmp_path: Path) -> None:
        env = resolved(tmp_path, "windows")
        data = json.loads(format_json(env))
        assert data["qairt_root"] == env.qairt_root.as_posix()
        assert data["platform"] == "windows"
        assert data["problems"] == []
        assert set(data["converters"]) == set(CONVERTER_TOOLS)


# ── pinned-config consistency (drift guards) ────────────────────────────────


class TestPinnedConfig:
    """These read the real repo files: they fail if the pins drift apart."""

    def test_versions_come_from_kavi_yaml(self) -> None:
        config = load_kavi_config()
        assert config.qairt_version == QAIRT_VERSION
        assert config.ndk_version == NDK_VERSION
        assert config.qairt_env_var == "QAIRT_SDK_ROOT"
        assert config.ndk_env_var == "ANDROID_NDK_ROOT"
        assert KAVI_CONFIG_PATH == REPO_ROOT / ".kavi.yaml"

    def test_converter_requirements_match_pyproject(self) -> None:
        """``scripts.env`` keeps the list for 3.10 compat; pyproject owns it."""
        with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
            qairt_group = tomllib.load(fh)["dependency-groups"]["qairt"]
        assert set(qairt_group) == set(CONVERTER_REQUIREMENTS)

    def test_kavi_env_example_paths_match_the_pins(self) -> None:
        """The example file and .kavi.yaml must not drift to different versions."""
        text = (REPO_ROOT / ".kavi.env.example").read_text(encoding="utf-8")
        qairt = re.search(r"QAIRT_SDK_ROOT=.*?/QAIRT/([0-9.]+)\}", text)
        ndk = re.search(r"ANDROID_NDK_ROOT=.*?/ndk/([0-9.]+)\}", text)
        assert qairt is not None and qairt.group(1) == QAIRT_VERSION
        assert ndk is not None and ndk.group(1) == NDK_VERSION
