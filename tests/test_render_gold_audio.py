"""Tests for scripts.render_gold_audio on synthetic files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.render_gold_audio import parse_snr, pick, readable, render_row


def _wav(path: Path, n: int, amp: float, seed: int) -> Path:
    x = amp * np.random.default_rng(seed).standard_normal(n)
    sf.write(str(path), x.astype("float32"), 16000)
    return path


@pytest.fixture
def assets(tmp_path: Path) -> dict:
    return {
        "clip": _wav(tmp_path / "clip.wav", 16000, 0.004, 1),
        "steady": [_wav(tmp_path / f"s{i}.wav", 8000, 0.05, 10 + i) for i in range(3)],
        "impulsive": [
            _wav(tmp_path / f"i{i}.wav", 8000, 0.05, 20 + i) for i in range(3)
        ],
        "rirs": [_wav(tmp_path / f"r{i}.wav", 800, 0.2, 30 + i) for i in range(3)],
    }


def test_parse_snr() -> None:
    assert parse_snr("clean") is None
    assert parse_snr("m5") == -5.0
    assert parse_snr("15") == 15.0
    assert parse_snr("0") == 0.0


def test_pick_wraps_around() -> None:
    files = [Path("a"), Path("b"), Path("c")]
    assert [pick(files, k).name for k in range(5)] == ["a", "b", "c", "a", "b"]


def test_clean_quiet_row_has_no_noise_or_rir(assets: dict) -> None:
    row = {"audio_path": str(assets["clip"]), "condition": "quiet", "snr_db": "clean"}
    built, meta = render_row(
        row, 0, assets["steady"], assets["impulsive"], assets["rirs"]
    )
    assert meta["noise"] is None and meta["rir"] is None
    assert len(built.audio) == 8000 + 16000 + 8000


def test_far_field_row_uses_impulsive_noise_and_rir(assets: dict) -> None:
    row = {"audio_path": str(assets["clip"]), "condition": "far-field", "snr_db": "m5"}
    _, meta = render_row(row, 4, assets["steady"], assets["impulsive"], assets["rirs"])
    assert Path(meta["noise"]).name == "i1.wav"
    assert Path(meta["rir"]).name == "r1.wav"
    assert meta["snr_db"] == -5.0


def test_indoors_row_uses_steady_noise_and_rir(assets: dict) -> None:
    row = {"audio_path": str(assets["clip"]), "condition": "indoors", "snr_db": "10"}
    _, meta = render_row(row, 2, assets["steady"], assets["impulsive"], assets["rirs"])
    assert Path(meta["noise"]).name == "s2.wav"
    assert meta["rir"] is not None


def test_street_row_has_no_rir(assets: dict) -> None:
    row = {"audio_path": str(assets["clip"]), "condition": "street", "snr_db": "5"}
    _, meta = render_row(row, 0, assets["steady"], assets["impulsive"], assets["rirs"])
    assert Path(meta["noise"]).name == "i0.wav"
    assert meta["rir"] is None


def test_readable_skips_unreadable_and_empty_files(
    tmp_path: Path, assets: dict
) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is not a wav file")
    empty = tmp_path / "empty.wav"
    sf.write(str(empty), np.zeros(0, dtype="float32"), 16000)
    good = assets["rirs"]
    assert readable([good[0], bad, empty, good[1]]) == [good[0], good[1]]
