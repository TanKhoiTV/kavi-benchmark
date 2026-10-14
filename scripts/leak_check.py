"""Assert the public boundary: no private-track artifacts in this repository.

The project is split across three repositories: this umbrella repo and
``kavi-prototype`` (now the host-side benchmark) are public, while the on-device
Android track (``kavi-android``) is private. This check is what *measures* that
boundary, so it runs in CI: a file whose type belongs to the private track cannot
reach ``main``, and the private repo cannot become a submodule of a public one.

It scans the **tracked** file list -- exactly what a clone receives -- and
reports three kinds of finding:

1. Private artifact types: native QNN libraries, packaged Android artifacts,
   Gradle/Android project files and app sources.
2. Material that is fetched rather than published: model weights, device-compiled
   models, and machine-local configuration.
3. Signing and credential material.

It also asserts the structural invariant the split calls out: no gitlink
(submodule entry) points at the private repository. A gitlink to a *public*
sibling is legitimate -- the umbrella repo tracks the benchmark as ``prototype/``
-- so allowed paths are passed in explicitly with ``--allow-gitlink`` rather than
being baked into the rules; each repo supplies its own list.

The same file is used by ``kavi-prototype``, and is stdlib-only and
self-contained on purpose so the two copies stay identical.

Exit status is non-zero when anything is found.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Index mode of a gitlink: a commit object recorded where a tree entry would be.
GITLINK_MODE = "160000"

#: Prefix of a ``.gitmodules`` section header.
SUBMODULE_SECTION = "[submodule"


class Rule(NamedTuple):
    """One thing that must not be published, and why."""

    id: str
    why: str
    suffixes: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    globs: tuple[str, ...] = ()


# Rules are grounded in the private repository's actual inventory: 39 ``.so``
# QNN libraries under ``app/src/main/jniLibs/arm64-v8a/``, Kotlin app sources,
# ``app/build.gradle.kts`` and ``app/src/main/AndroidManifest.xml``. Anything
# fetched by ``make models`` / ``make data`` is pinned in ``assets.lock.toml``
# instead of committed, so a tracked weight file is a leak by definition.
RULES: tuple[Rule, ...] = (
    Rule(
        "native-library",
        "native binary - QNN libraries belong to the private app track",
        suffixes=(".so", ".dll", ".dylib"),
    ),
    Rule(
        "android-package",
        "packaged Android artifact - publish the app, not the artifact",
        suffixes=(".apk", ".aab", ".aar", ".jar"),
    ),
    Rule(
        "android-project",
        "Gradle/Android build file or manifest - the app track owns these",
        suffixes=(".kts", ".gradle"),
        names=("androidmanifest.xml", "gradlew", "gradlew.bat"),
    ),
    Rule(
        "app-source",
        "app source language - the on-device track owns Kotlin/C++/Java",
        suffixes=(".kt", ".java", ".cpp", ".cc", ".hpp"),
    ),
    Rule(
        "compiled-model",
        "device-compiled model - regenerate it, do not commit it",
        suffixes=(".dlc", ".qnn", ".tflite"),
    ),
    Rule(
        "model-weights",
        "model weights - fetch via `make models`, pin in assets.lock.toml",
        suffixes=(".bin", ".onnx", ".pt", ".pth", ".ckpt", ".gguf", ".safetensors"),
    ),
    Rule(
        "secret-material",
        "signing or credential material - never publish this",
        suffixes=(".jks", ".keystore", ".p12", ".pfx", ".pem", ".key", ".pk8"),
        names=("keystore.properties", ".netrc", "id_rsa", "id_ed25519"),
    ),
    Rule(
        "machine-local",
        "machine-local config - documents the shape, not the real paths",
        names=(".kavi.env", "local.properties"),
    ),
    Rule(
        "on-device-result",
        "on-device run output - on-device numbers stay in the private repo",
        globs=("*run_results_android*.json",),
    ),
)


class Finding(NamedTuple):
    """A tracked path that must not be public, plus the rule it broke."""

    rule: str
    path: str
    why: str


def classify(path: str) -> Rule | None:
    """Return the first rule *path* breaks, or ``None`` when it is publishable.

    *path* is repository-relative and POSIX-spelled. Suffix and basename matching
    is case-insensitive; globs are matched against the whole path with
    ``fnmatch``, so ``*run_results_android*.json`` covers any directory.
    """
    lowered = path.lower()
    name = PurePosixPath(lowered).name
    for rule in RULES:
        if rule.suffixes and lowered.endswith(rule.suffixes):
            return rule
        if name in rule.names:
            return rule
        if any(fnmatch.fnmatchcase(lowered, pattern) for pattern in rule.globs):
            return rule
    return None


def scan(paths: Iterable[str]) -> list[Finding]:
    """Classify every path, returning findings sorted by path."""
    found = [
        Finding(rule.id, path, rule.why)
        for path in paths
        if (rule := classify(path)) is not None
    ]
    return sorted(found, key=lambda finding: (finding.path, finding.rule))


def gitlink_findings(
    index_entries: Iterable[str], allowed: Collection[str] = ()
) -> list[Finding]:
    """Find gitlinks in ``git ls-files -s`` output, split on NUL or newlines.

    Each entry reads ``<mode> <object> <stage>\\t<path>``. A gitlink is what makes
    a submodule. Paths in *allowed* are public siblings this repo may track.
    """
    found: list[Finding] = []
    for entry in index_entries:
        entry = entry.strip()
        if not entry:
            continue
        mode, _, rest = entry.partition(" ")
        if mode != GITLINK_MODE:
            continue
        path = rest.rpartition("\t")[2] or rest
        if path in allowed:
            continue
        found.append(
            Finding(
                "gitlink",
                path,
                "gitlink (submodule) entry - the private repo is never a submodule",
            )
        )
    return sorted(found, key=lambda finding: finding.path)


def submodule_entries(text: str) -> list[tuple[str, str]]:
    """Return ``(section header, path)`` for every section in ``.gitmodules``.

    Sections without a ``path`` key are skipped -- a gitlink in the index is what
    makes a submodule, and that is checked separately.
    """
    entries: list[tuple[str, str]] = []
    header = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(SUBMODULE_SECTION):
            header = stripped
            continue
        if not header or not stripped.lower().startswith("path"):
            continue
        _, _, value = stripped.partition("=")
        entries.append((header, value.strip()))
        header = ""
    return entries


def submodule_sections(text: str) -> list[str]:
    """Return the ``.gitmodules`` section headers, if that file has any."""
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip().lower().startswith(SUBMODULE_SECTION)
    ]


def _git(repo_root: Path, *args: str) -> str:
    """Run a git query in *repo_root*, with actionable failures."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "git is not on PATH - this check reads the tracked file list"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}") from exc
    return result.stdout


def tracked_paths(repo_root: Path) -> list[str]:
    """Return the POSIX-spelled tracked file list for *repo_root*."""
    output = _git(repo_root, "ls-files", "-z")
    return [path for path in output.split("\0") if path]


def index_entries(repo_root: Path) -> list[str]:
    """Return ``git ls-files -s -z`` entries for *repo_root*."""
    output = _git(repo_root, "ls-files", "-s", "-z")
    return [entry for entry in output.split("\0") if entry]


def check(
    repo_root: Path, allowed_gitlinks: Collection[str] = ()
) -> tuple[list[str], list[Finding]]:
    """Run every boundary check, returning the scanned paths and all findings."""
    paths = tracked_paths(repo_root)
    found = scan(paths)
    found += gitlink_findings(index_entries(repo_root), allowed_gitlinks)
    gitmodules = repo_root / ".gitmodules"
    if gitmodules.is_file():
        text = gitmodules.read_text(encoding="utf-8")
        found += [
            Finding("gitmodules", entry_path, f"declares a submodule: {header}")
            for header, entry_path in submodule_entries(text)
            if entry_path not in allowed_gitlinks
        ]
    return paths, sorted(found, key=lambda finding: (finding.path, finding.rule))


def _report(paths: Sequence[str], found: Sequence[Finding]) -> None:
    print("Public boundary check")
    print(f"\n  files scanned : {len(paths)}")
    print(f"  findings      : {len(found)}")
    if not found:
        print("\n  OK   no private-track artifacts - the public boundary holds.")
        return
    print("\n  FAIL")
    for finding in found:
        print(f"    {finding.rule:16} {finding.path}")
        print(f"    {'':16} {finding.why}")
    print("\n  1 or more tracked files belong to the private track.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository to inspect (default: this file's repository).",
    )
    parser.add_argument(
        "--allow-gitlink",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Submodule path that is a public sibling and therefore allowed "
            "(repeatable). Without it, any gitlink is a finding."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Explicit POSIX-spelled paths to classify instead of reading the "
            "index. Useful for a hook; the default scans the whole repository."
        ),
    )
    args = parser.parse_args(argv)

    try:
        if args.paths:
            paths = list(args.paths)
            found = scan(paths)
        else:
            paths, found = check(args.repo_root, args.allow_gitlink)
    except RuntimeError as exc:
        print(f"[leak-check] {exc}", file=sys.stderr)
        return 2

    _report(paths, found)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
