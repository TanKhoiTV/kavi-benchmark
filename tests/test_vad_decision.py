"""Tests for bench.vad_decision on a synthetic item with a known onset delay."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bench import vad_decision as vd
from bench.vad_manifest import VadItem, VadManifest

SR = 16000
LEAD_IN = 1920


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
    paths = []
    for name, x in (("a", a), ("b", b)):
        p = tmp_path / f"{name}.wav"
        sf.write(str(p), x.astype("float32"), SR)
        paths.append(str(p))
    b0 = 32000 + 200 * 16
    return VadItem(
        id="delayed",
        language="vi",
        condition="quiet",
        clean_audio_path=paths[0],
        noise_type="clean",
        speech_b_path=paths[1],
        pause_ms=200,
        ground_truth_segments=[(16000, 32000), (b0, b0 + 16000)],
    )


def test_cell_statuses_follow_the_borderline_rule() -> None:
    clean = {
        "missed_onset_rate": 0.0,
        "false_trigger_rate_per_min": 0.0,
        "clipped_ms_per_utterance": 10.0,
    }
    assert vd.all_pass(vd.cell_statuses(clean))
    border = {**clean, "clipped_ms_per_utterance": 55.0}
    assert not vd.all_pass(vd.cell_statuses(border)) and not vd.has_fail(
        vd.cell_statuses(border)
    )
    failing = {**clean, "missed_onset_rate": 0.4}
    assert vd.has_fail(vd.cell_statuses(failing))


def test_label_error_sensitivity_is_flagged_without_deciding(
    delayed_item: VadItem,
) -> None:
    rows, summaries = vd.evaluate_pairs([delayed_item], [0.05], [500], 0.125, 0.025)
    assert len(rows) == 1
    row = rows[0]
    assert row["all_pass"] is True
    assert row["label_error_sensitive"] is True
    assert summaries[(0.05, 500)]["label_sensitive_cells"] == 1


def test_a_cell_far_from_the_boundary_is_not_flagged(delayed_item: VadItem) -> None:
    rows, _ = vd.evaluate_pairs([delayed_item], [0.05], [500], 0.20, 0.02)
    assert rows[0]["all_pass"] is True
    assert rows[0]["label_error_sensitive"] is False


def test_small_collar_fails_and_is_counted(delayed_item: VadItem) -> None:
    rows, summaries = vd.evaluate_pairs([delayed_item], [0.05], [500], 0.03, 0.0)
    assert rows[0]["all_pass"] is False
    assert summaries[(0.05, 500)]["failing_cells"] == 1
    assert summaries[(0.05, 500)]["fails_by_criterion"]["missed_onset_rate"] == 1
    assert not summaries[(0.05, 500)]["passes_everywhere"]


def test_decide_case() -> None:
    assert vd.decide_case({(0.05, 500): {"passes_everywhere": False}}) == ("B", [])
    case, passing = vd.decide_case(
        {
            (0.05, 500): {"passes_everywhere": False},
            (0.1, 350): {"passes_everywhere": True},
        }
    )
    assert case == "A" and passing == [(0.1, 350)]


def test_failing_cells_by_condition_counts_only_failing_cells() -> None:
    def row(cond, snr, passing, miss):
        return {
            "threshold": 0.05, "timeout_ms": 500, "condition": cond, "snr_db": snr,
            "all_pass": passing, "status_missed_onset_rate": miss,
            "status_false_trigger_rate_per_min": "pass", "status_clipped_ms_per_utterance": "pass",
        }  # fmt: skip

    rows = [
        row("quiet", "clean", True, "pass"),
        row("street", 10.0, False, "fail"),
        row("street", 10.0, False, "borderline"),
    ]
    out = vd.failing_cells_by_condition(rows, (0.05, 500))
    assert list(out) == ["street/10.0"]
    assert (
        out["street/10.0"]["cells"] == 2
        and out["street/10.0"]["missed_onset_rate"] == 1
    )


def test_main_refuses_without_collar(tmp_path: Path) -> None:
    rc = vd.main(
        ["--collar-file", str(tmp_path / "missing.json"), "--out", str(tmp_path / "o")]
    )
    assert rc == 1 and not (tmp_path / "o").exists()


def test_main_writes_only_under_the_output_directory(
    delayed_item: VadItem, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "m.json"
    VadManifest(items=[delayed_item]).to_json(manifest)
    collar = tmp_path / "collar.json"
    collar.write_text(
        json.dumps({"collar_s": 0.125, "partial": False}), encoding="utf-8"
    )
    out = tmp_path / "out"
    rc = vd.main(
        [
            "--manifest",
            str(manifest),
            "--collar-file",
            str(collar),
            "--label-error-ms",
            "25",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    text = capsys.readouterr().out
    assert "Pairs passing in all 1 cells" in text
    assert "cells sensitive to the label error" in text
    with open(out / "cells.csv", encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 25
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert data["case"] in {"A", "B"} and len(data["pairs"]) == 25
