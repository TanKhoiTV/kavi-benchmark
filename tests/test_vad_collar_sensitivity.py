"""Tests for bench.vad_collar_sensitivity on synthetic WAVs with a known onset delay."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bench.vad_collar_sensitivity import (
    CSV_FIELDS,
    condition_rows,
    first_collar_meeting,
    main,
    meets_all,
    run_collar_sensitivity,
)
from bench.vad_manifest import VadItem, VadManifest
from bench.vad_runner import run_config, summarize

SR = 16000
LEAD_IN = 1920


def _write(path: Path, x: np.ndarray) -> str:
    sf.write(str(path), x.astype("float32"), SR)
    return str(path)


@pytest.fixture
def delayed_item(tmp_path: Path) -> VadItem:
    rng = np.random.default_rng(7)
    a = np.concatenate(
        [
            0.0005 * rng.standard_normal(LEAD_IN),
            0.3 * rng.standard_normal(16000 - LEAD_IN),
        ]
    )
    b = 0.3 * rng.standard_normal(16000)
    pause = 200
    b0 = 32000 + pause * 16
    return VadItem(
        id="delayed",
        language="vi",
        condition="quiet",
        clean_audio_path=_write(tmp_path / "a.wav", a),
        noise_type="clean",
        speech_b_path=_write(tmp_path / "b.wav", b),
        pause_ms=pause,
        ground_truth_segments=[(16000, 32000), (b0, b0 + 16000)],
    )


def test_collar_decides_missed_onset_and_clipping(delayed_item: VadItem) -> None:
    results = run_collar_sensitivity([delayed_item], [0.05], [500], [0, 100, 150])
    zero = results[(0.05, 500, 0)][0].metrics
    mid = results[(0.05, 500, 100)][0].metrics
    wide = results[(0.05, 500, 150)][0].metrics
    assert (
        zero.missed_onset_count,
        mid.missed_onset_count,
        wide.missed_onset_count,
    ) == (1, 1, 0)
    assert zero.clipped_start_ms == pytest.approx(120.0, abs=1.0)
    assert mid.clipped_start_ms == pytest.approx(20.0, abs=1.0)
    assert wide.clipped_start_ms == 0.0


def test_results_never_get_worse_with_a_larger_collar(delayed_item: VadItem) -> None:
    collars = [0, 25, 50, 75, 100, 125, 150, 200]
    results = run_collar_sensitivity([delayed_item], [0.05], [500], collars)
    missed = [results[(0.05, 500, c)][0].metrics.missed_onset_count for c in collars]
    clipped = [results[(0.05, 500, c)][0].metrics.clipped_start_ms for c in collars]
    assert missed == sorted(missed, reverse=True)
    assert clipped == sorted(clipped, reverse=True)


def test_matches_single_config_runner_at_the_same_collar(delayed_item: VadItem) -> None:
    results = run_collar_sensitivity([delayed_item], [0.05, 0.2], [200, 500], [150])
    for thr in (0.05, 0.2):
        for to in (200, 500):
            single, _ = run_config([delayed_item], thr, to, 0.150)
            assert summarize(results[(thr, to, 150)]) == summarize(single)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({0: False, 50: False, 100: True, 150: True}, 100),
        ({0: True, 50: False, 100: True, 150: True}, 100),
        ({0: False, 50: False}, None),
        ({0: True, 50: True}, 0),
    ],
)
def test_first_collar_meeting(flags: dict[int, bool], expected: int | None) -> None:
    assert first_collar_meeting(flags) == expected


def test_meets_all_requires_every_criterion() -> None:
    cell = {
        "missed_onset_rate": 0.0,
        "false_trigger_rate_per_min": 0.0,
        "clipped_ms_per_utterance": {"mean": 10.0},
    }
    assert meets_all(cell)
    assert not meets_all({**cell, "missed_onset_rate": 0.2})
    assert not meets_all({**cell, "false_trigger_rate_per_min": 1.0})
    assert not meets_all({**cell, "clipped_ms_per_utterance": {"mean": 80.0}})
    assert not meets_all({**cell, "false_trigger_rate_per_min": None})


def test_condition_rows_cover_every_config_and_collar(delayed_item: VadItem) -> None:
    results = run_collar_sensitivity([delayed_item], [0.05, 0.2], [200, 500], [0, 150])
    rows = condition_rows(results)
    assert len(rows) == 2 * 2 * 2
    assert set(rows[0]) == set(CSV_FIELDS)


def test_main_writes_csv_and_report(
    delayed_item: VadItem, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "m.json"
    VadManifest(items=[delayed_item]).to_json(manifest)
    out = tmp_path / "out"
    rc = main(
        [
            "--manifest", str(manifest),
            "--thresholds", "0.05,0.2",
            "--timeouts", "200,500",
            "--collars-ms", "0,100,150",
            "--out", str(out),
        ]
    )  # fmt: skip
    assert rc == 0
    text = capsys.readouterr().out
    assert "missed-onset %" in text and "Smallest collar" in text
    with open(out / "sensitivity.csv", encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 2 * 2 * 3
    data = json.loads((out / "sensitivity.json").read_text(encoding="utf-8"))
    assert data["collars_ms"] == [0, 100, 150]
