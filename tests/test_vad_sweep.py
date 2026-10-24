"""Tests for bench.vad_sweep on small synthetic WAVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bench.vad_manifest import VadItem, VadManifest
from bench.vad_runner import run_config, summarize
from bench.vad_sweep import (
    CSV_FIELDS,
    cell_assessments,
    main,
    run_sweep,
    summarize_config,
)

SR = 16000


@pytest.fixture
def items(tmp_path: Path) -> list[VadItem]:
    rng = np.random.default_rng(1)
    paths = []
    for name in ("a", "b"):
        p = tmp_path / f"{name}.wav"
        sf.write(str(p), (0.3 * rng.standard_normal(16000)).astype("float32"), SR)
        paths.append(str(p))
    out = []
    for pause in (200, 350, 500, 700):
        for lang in ("vi", "en"):
            b0 = 32000 + pause * 16
            out.append(
                VadItem(
                    id=f"{lang}-{pause}",
                    language=lang,
                    condition="quiet",
                    clean_audio_path=paths[0],
                    noise_type="clean",
                    speech_b_path=paths[1],
                    pause_ms=pause,
                    ground_truth_segments=[(16000, 32000), (b0, b0 + 16000)],
                )
            )
    return out


def test_sweep_matches_single_config_runner(items: list[VadItem]) -> None:
    grid_t, grid_to = [0.05, 0.2], [200, 500]
    swept, _ = run_sweep(items, grid_t, grid_to, 0.150)
    for thr in grid_t:
        for to in grid_to:
            single, _ = run_config(items, thr, to, 0.150)
            assert summarize(swept[(thr, to)]) == summarize(single)


def test_sweep_changes_with_timeout(items: list[VadItem]) -> None:
    swept, _ = run_sweep(items, [0.05], [200, 1000], 0.150)
    short = summarize(swept[(0.05, 200)])
    long_ = summarize(swept[(0.05, 1000)])
    assert short["junction_split_rate_by_pause"]["700"] == 1.0
    assert long_["junction_split_rate_by_pause"]["700"] == 0.0


def test_cell_assessments_counts_every_cell(items: list[VadItem]) -> None:
    results, _ = run_sweep(items, [0.05], [500], 0.150)
    info = cell_assessments(results[(0.05, 500)])
    assert info["n_cells"] == 4
    assert info["items_per_lang_per_cell"] == [1]
    for counts in info["counts"].values():
        assert sum(counts.values()) == 4


def test_summarize_config_reports_false_trigger_bound(items: list[VadItem]) -> None:
    results, _ = run_sweep(items, [0.05], [500], 0.150)
    summary = summarize_config(results[(0.05, 500)])
    assert summary["overall"]["false_trigger_count"] == 0
    assert summary["false_trigger_upper95_per_min"] == pytest.approx(
        3.0 / summary["non_speech_minutes"]
    )


def test_main_writes_json_and_csv(
    items: list[VadItem], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "m.json"
    VadManifest(items=items).to_json(manifest)
    out = tmp_path / "out"
    rc = main(
        [
            "--manifest", str(manifest),
            "--thresholds", "0.05,0.2",
            "--timeouts", "200,500",
            "--collar-file", str(tmp_path / "missing.json"),
            "--out", str(out),
        ]
    )  # fmt: skip
    assert rc == 0
    text = capsys.readouterr().out
    assert "PROVISIONAL" in text and "missed-onset rate %" in text
    data = json.loads((out / "sweep.json").read_text(encoding="utf-8"))
    assert len(data["configs"]) == 4
    with open(out / "sweep.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0]) == set(CSV_FIELDS)
    assert len(rows) == 4 * 2
