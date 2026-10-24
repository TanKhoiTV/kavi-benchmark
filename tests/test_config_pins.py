"""Drift guards for the published pin file (``.kavi.yaml``).

The file holds the published gates and claims; the on-device mechanism lives in
``kavi-android/kavi-tuning.yaml``. These tests keep the two boundaries honest
without needing the private repository: they assert what the public file owns,
that the host pin matches ``pyproject.toml``, and that no on-device tuning key
has crept back in.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Top-level keys the published file is allowed to carry. Adding one is a
#: decision about the public claim, so it belongs in a review, not a drive-by.
PUBLISHED_KEYS = {
    "metadata",
    "asr",
    "mt",
    "tts",
    "denoiser",
    "memory",
    "timing",
    "runtime",
}


def read_published() -> str:
    return (REPO_ROOT / ".kavi.yaml").read_text(encoding="utf-8")


def test_published_sherpa_onnx_host_pin_matches_pyproject() -> None:
    """``.kavi.yaml`` records the host pin for provenance; pyproject owns it.

    The on-device AAR pin is a different artifact and lives in
    ``kavi-android/kavi-tuning.yaml`` — it is deliberately not compared here.
    """
    text = read_published()
    published = re.search(
        r"^\s*sherpa_onnx_host:\s*\"?([^\"\s#]+)\"?", text, re.MULTILINE
    )
    assert published is not None, "runtime.sherpa_onnx_host is missing"
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        dependencies = tomllib.load(fh)["project"]["dependencies"]
    pinned = [dep for dep in dependencies if dep.startswith("sherpa-onnx==")]
    assert len(pinned) == 1, pinned
    assert published.group(1) == pinned[0].split("==", 1)[1]


def test_only_published_keys_are_present() -> None:
    """No on-device tuning key may reappear in the public file."""
    keys = {
        match.group(1)
        for line in read_published().splitlines()
        if (match := re.match(r"^([a-z_]+):", line))
    }
    assert keys == PUBLISHED_KEYS, f"unexpected top-level keys: {keys ^ PUBLISHED_KEYS}"


def test_mechanism_keys_do_not_reappear() -> None:
    """Spot-check the values the split moved private."""
    text = read_published()
    for leaked in (
        "num_threads:",
        "vad_speech_timeout_ms:",
        "jni_libs_trimmed:",
        "qairt:",
    ):
        assert leaked not in text, (
            f"{leaked} is on-device mechanism, not a published claim"
        )
