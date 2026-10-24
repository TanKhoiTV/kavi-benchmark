"""Tests for the public-boundary check (``scripts/leak_check.py``).

The path fixtures are the private repository's real inventory, so a rule change
that stops covering, say, ``libQnnHtp.so`` or ``app/build.gradle.kts`` fails here
rather than after the files are already public.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.leak_check import (
    REPO_ROOT,
    check,
    classify,
    gitlink_findings,
    main,
    scan,
    submodule_entries,
    submodule_sections,
    tracked_paths,
)

#: Representative private-track paths, mapped to the rule that must catch them.
PRIVATE_PATHS = {
    "app/src/main/jniLibs/arm64-v8a/libQnnHtp.so": "native-library",
    "app/src/main/jniLibs/arm64-v8a/libQnnDspV66Stub.so": "native-library",
    "qnn/2.31.0/x86_64-windows-msvc/QnnHtp.dll": "native-library",
    "app/build.gradle.kts": "android-project",
    "build.gradle": "android-project",
    "settings.gradle": "android-project",
    "app/src/main/AndroidManifest.xml": "android-project",
    "gradlew": "android-project",
    "app/src/main/java/com/kavi/app/qnn/QnnModelLoader.kt": "app-source",
    "app/src/main/cpp/qnn_loader_jni.cpp": "app-source",
    "gradle/wrapper/gradle-wrapper.jar": "android-package",
    "release/kavi-arm64.apk": "android-package",
    "build/outputs/kavi.aar": "android-package",
    "models/whisper_encoder.dlc": "compiled-model",
    "models/opus_mt_ctx.qnn": "compiled-model",
    "models/encoder.tflite": "compiled-model",
    "models/pytorch_model.bin": "model-weights",
    "models/encoder_fp16.onnx": "model-weights",
    "models/weights.safetensors": "model-weights",
    "release/kavi-release.jks": "secret-material",
    "release/upload-keystore.p12": "secret-material",
    "deploy/id_rsa": "secret-material",
    ".kavi.env": "machine-local",
    "android/local.properties": "machine-local",
    "bench-results/run_results_android.json": "on-device-result",
    "runs/2026-10/run_results_android_v1.json": "on-device-result",
    "bench-results/RUN_RESULTS_ANDROID.json": "on-device-result",
}

#: Paths that must stay publishable -- the false-positive guard. ``.kavi.env``
#: is banned while its tracked ``.example`` twin is the documented workflow.
PUBLIC_PATHS = (
    "README.md",
    "CONTRIBUTING.md",
    "NOTICE",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "cliff.toml",
    "commitlint.config.js",
    "assets.lock.toml",
    ".gitattributes",
    ".gitignore",
    ".kavi.env.example",
    "models/.gitkeep",
    "scripts/leak_check.py",
    "scripts/env.py",
    "scripts/qairt-env.sh",
    "scripts/qairt-env.ps1",
    "tests/test_leak_check.py",
    "docs/README.md",
    "docs/ndk-conversion-runbook.md",
    "docs/reference/phase-4-qnn-plan.md",
    "bench/qnn/convert_to_qnn.sh",
    "bench/run.py",
    ".github/workflows/ci.yml",
    ".github/CODEOWNERS",
    "archive/pipeline.py",
    "eval_data/eval_manifest_v1.json",
    "models/README.md",
)


class TestClassify:
    """Rule matching: private paths are caught, public ones are not."""

    @pytest.mark.parametrize(("path", "rule_id"), sorted(PRIVATE_PATHS.items()))
    def test_private_path_is_caught(self, path: str, rule_id: str) -> None:
        rule = classify(path)
        assert rule is not None, f"{path} slipped through the boundary"
        assert rule.id == rule_id

    @pytest.mark.parametrize("path", PUBLIC_PATHS)
    def test_public_path_is_allowed(self, path: str) -> None:
        assert classify(path) is None, f"{path} is publishable"

    def test_kavi_env_example_is_not_the_banned_file(self) -> None:
        """``.kavi.env`` is machine-local; the tracked example must stay legal."""
        assert classify(".kavi.env") is not None
        assert classify(".kavi.env.example") is None

    def test_matching_is_case_insensitive(self) -> None:
        assert classify("APP/src/main/AndroidManifest.xml") is not None
        assert classify("Models/PyTorch_Model.BIN") is not None

    def test_every_rule_is_reachable(self) -> None:
        """A rule that matches nothing is dead weight and should be deleted."""
        hit = {finding.rule for finding in scan(PRIVATE_PATHS)}
        assert hit == set(PRIVATE_PATHS.values())


class TestScan:
    """``scan`` reports findings in a stable order."""

    def test_clean_input_has_no_findings(self) -> None:
        assert scan(PUBLIC_PATHS) == []

    def test_findings_are_sorted_by_path(self) -> None:
        found = scan(["z/libQnnHtp.so", "a/keystore.jks", "README.md"])
        assert [finding.path for finding in found] == [
            "a/keystore.jks",
            "z/libQnnHtp.so",
        ]

    def test_finding_keeps_the_rule_reason(self) -> None:
        (finding,) = scan(["app/build.gradle.kts"])
        assert finding.rule == "android-project"
        assert "Gradle" in finding.why


class TestGitlinks:
    """The split's structural invariant: no gitlink to the private repo."""

    def test_gitlink_is_reported(self) -> None:
        (finding,) = gitlink_findings(["160000 1a2b3c 0\tandroid"])
        assert finding.rule == "gitlink"
        assert finding.path == "android"

    def test_ordinary_modes_are_ignored(self) -> None:
        assert (
            gitlink_findings(["100644 abc 0\tREADME.md", "100755 def 0\trun.sh"]) == []
        )

    def test_blank_entries_are_ignored(self) -> None:
        assert gitlink_findings(["", "  "]) == []

    def test_multiple_gitlinks_are_sorted(self) -> None:
        found = gitlink_findings(["160000 a 0\tzz", "160000 b 0\taa"])
        assert [finding.path for finding in found] == ["aa", "zz"]

    def test_allow_list_only_covers_the_paths_given(self) -> None:
        """The umbrella repo allows its public submodule; the private one never is."""
        entries = ["160000 a 0\tprototype", "160000 b 0\tandroid"]
        found = gitlink_findings(entries, ["prototype"])
        assert [finding.path for finding in found] == ["android"]

    def test_submodule_entries_are_parsed(self) -> None:
        text = (
            "# a comment\n"
            '[submodule "prototype"]\n'
            "\tpath = prototype\n"
            "\turl = git@github.com:TanKhoiTV/kavi-prototype.git\n"
        )
        assert submodule_entries(text) == [('[submodule "prototype"]', "prototype")]

    def test_comment_only_gitmodules_declares_nothing(self) -> None:
        text = "# the app repo is deliberately NOT a submodule\n#\n# cli: clone it\n"
        assert submodule_sections(text) == []

    def test_submodule_section_is_detected(self) -> None:
        text = '[submodule "android"]\n\tpath = android\n\turl = git@example:x.git\n'
        assert submodule_sections(text) == ['[submodule "android"]']


class TestThisRepository:
    """The check's own assertion about the real tree -- what CI enforces."""

    def test_repository_is_within_the_boundary(self) -> None:
        paths, found = check(REPO_ROOT)
        assert found == [], f"private-track files are tracked: {found}"
        # A sanity floor, not a target: without it a broken `git ls-files` would
        # make this check pass vacuously. The 2026-10 re-organisation moved the
        # on-device track out, so the tree is ~96 tracked paths, not ~180.
        assert len(paths) > 50, "expected the real tracked file list"

    def test_the_scanned_list_is_the_git_index(self) -> None:
        """Untracked files are out of scope: the check asks what a clone receives."""
        paths, _ = check(REPO_ROOT)
        assert sorted(paths) == sorted(tracked_paths(REPO_ROOT))
        assert "pyproject.toml" in paths and "Makefile" in paths


class TestMain:
    """CLI behaviour: exit status and console-safe output."""

    def test_clean_repository_exits_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([]) == 0
        captured = capsys.readouterr()
        assert "OK" in captured.out
        assert captured.out.isascii()

    def test_allow_gitlink_flag_is_accepted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Unused here (there is no submodule), but it keeps both copies one file."""
        assert main(["--allow-gitlink", "prototype"]) == 0
        assert "OK" in capsys.readouterr().out

    def test_explicit_private_path_exits_nonzero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["app/src/main/jniLibs/arm64-v8a/libQnnHtp.so"]) == 1
        captured = capsys.readouterr()
        assert "native-library" in captured.out
        assert captured.out.isascii()

    def test_unreadable_repository_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "file.txt").write_text("not a repo\n", encoding="utf-8")
        assert main(["--repo-root", str(tmp_path)]) == 2
        captured = capsys.readouterr()
        assert "leak-check" in captured.err
        assert captured.err.isascii()
