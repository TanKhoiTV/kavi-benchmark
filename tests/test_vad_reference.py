"""Tests for bench.vad_reference (pure parts; Silero itself needs torch)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.vad_reference import load_source_bounds, outer_bounds


def test_outer_bounds_spans_all_timestamps() -> None:
    stamps = [{"start": 4000, "end": 9000}, {"start": 1500, "end": 3000}]
    assert outer_bounds(stamps) == (1500, 9000)


def test_outer_bounds_empty_is_none() -> None:
    assert outer_bounds([]) is None


def write_bounds(path: Path, files: dict) -> str:
    path.write_text(json.dumps({"files": files}), encoding="utf-8")
    return str(path)


def test_load_source_bounds_reads_valid_entries(tmp_path: Path) -> None:
    p = write_bounds(
        tmp_path / "b.json", {"a.wav": {"onset": 100, "offset": 900, "n_samples": 1000}}
    )
    assert load_source_bounds(p) == {"a.wav": (100, 900)}


@pytest.mark.parametrize(
    "entry",
    [
        {"onset": 900, "offset": 100, "n_samples": 1000},
        {"onset": 0, "offset": 2000, "n_samples": 1000},
        {"onset": -1, "offset": 10, "n_samples": 1000},
    ],
)
def test_load_source_bounds_rejects_invalid_entries(
    tmp_path: Path, entry: dict
) -> None:
    with pytest.raises(ValueError):
        load_source_bounds(write_bounds(tmp_path / "b.json", {"a.wav": entry}))


def test_load_source_bounds_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_source_bounds(str(tmp_path / "none.json"))
