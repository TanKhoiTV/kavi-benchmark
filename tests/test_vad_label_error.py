"""Tests for the Stage 3B comparison functions on hand-made boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench import vad_label_error as vle
from bench.vad_runner import COLLAR_PATH as RUNNER_COLLAR_PATH

REFERENCE = {"a": (1.00, 3.00), "b": (2.00, 4.00), "c": (0.50, 1.50)}
HAND = {"a": (1.05, 2.98), "b": (2.00, 4.20), "c": (0.60, 1.50)}


def test_nearest_rank_uses_ceil_rank() -> None:
    values = [0.0, 0.0, 0.02, 0.05, 0.1, 0.2]
    assert vle.nearest_rank(values, 95) == 0.2
    assert vle.nearest_rank(values, 50) == 0.02
    assert vle.nearest_rank(values, 100) == 0.2
    assert vle.nearest_rank(list(range(1, 101)), 95) == 95


def test_nearest_rank_rejects_empty() -> None:
    with pytest.raises(ValueError):
        vle.nearest_rank([], 95)


def test_boundary_errors_are_reference_minus_hand() -> None:
    errors = vle.boundary_errors(REFERENCE, HAND)
    assert errors["ids"] == ["a", "b", "c"]
    assert errors["onset"] == pytest.approx([-0.05, 0.0, -0.10])
    assert errors["offset"] == pytest.approx([0.02, -0.20, 0.0])


def test_boundary_errors_ignore_ids_missing_on_either_side() -> None:
    errors = vle.boundary_errors({**REFERENCE, "x": (0.0, 1.0)}, HAND)
    assert errors["ids"] == ["a", "b", "c"]


def test_summarize_onset_and_offset_are_separate() -> None:
    summary = vle.summarize(vle.boundary_errors(REFERENCE, HAND))
    assert summary["onset"]["signed"]["mean"] == pytest.approx(-0.05)
    assert summary["onset"]["signed"]["median"] == pytest.approx(-0.05)
    assert summary["onset"]["absolute"]["mean"] == pytest.approx(0.05)
    assert summary["offset"]["signed"]["median"] == pytest.approx(0.0)
    assert summary["offset"]["abs_p95"] == pytest.approx(0.20)
    assert summary["pooled_abs"]["n"] == 6


def test_collar_is_p95_of_pooled_absolute_errors() -> None:
    errors = vle.boundary_errors(REFERENCE, HAND)
    assert vle.derive_collar(errors) == pytest.approx(0.20)
    assert vle.derive_collar(errors) == vle.summarize(errors)["pooled_abs"]["p95"]


def test_collar_with_many_clips_matches_rank_formula() -> None:
    ref = {str(i): (1.0, 2.0) for i in range(50)}
    hand = {str(i): (1.0 + i / 1000.0, 2.0) for i in range(50)}
    errors = vle.boundary_errors(ref, hand)
    pooled = sorted([i / 1000.0 for i in range(50)] + [0.0] * 50)
    assert vle.derive_collar(errors) == pytest.approx(pooled[94])


def test_stage1_comparison_flags_each_limit_separately() -> None:
    assert vle.stage1_comparison(0.04) == {
        "onset_tolerance_s": 0.15,
        "clipping_limit_s": 0.05,
        "collar_exceeds_onset_tolerance": False,
        "collar_exceeds_clipping_limit": False,
    }
    mid = vle.stage1_comparison(0.10)
    assert (
        mid["collar_exceeds_clipping_limit"]
        and not mid["collar_exceeds_onset_tolerance"]
    )
    high = vle.stage1_comparison(0.20)
    assert (
        high["collar_exceeds_clipping_limit"] and high["collar_exceeds_onset_tolerance"]
    )


def test_load_labels_skips_blank_rows_and_validates(tmp_path: Path) -> None:
    f = tmp_path / "labels.csv"
    f.write_text(
        "id,audio_path,onset_s,offset_s,annotator\ng1,a.wav,0.8,3.5,me\ng2,b.wav,,,\n",
        encoding="utf-8",
    )
    labels, blank = vle.load_labels(str(f))
    assert labels == {"g1": (0.8, 3.5)}
    assert blank == 1
    f.write_text(
        "id,audio_path,onset_s,offset_s,annotator\ng1,a.wav,3.0,1.0,me\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        vle.load_labels(str(f))


def test_main_refuses_incomplete_labels(tmp_path: Path) -> None:
    f = tmp_path / "labels.csv"
    f.write_text(
        "id,audio_path,onset_s,offset_s,annotator\ng1,a.wav,,,\n", encoding="utf-8"
    )
    assert vle.main(["--labels", str(f), "--out", str(tmp_path / "c.json")]) == 1
    assert not (tmp_path / "c.json").exists()


def test_collar_path_matches_runner() -> None:
    assert vle.COLLAR_PATH == RUNNER_COLLAR_PATH


def test_main_writes_collar_file_that_the_runner_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import numpy as np
    import soundfile as sf

    from bench import vad_reference
    from bench.vad_runner import load_collar

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for clip_id in ("g1", "g2", "g3"):
        sf.write(str(audio_dir / f"{clip_id}.wav"), np.zeros(16000, "float32"), 16000)
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "id,audio_path,onset_s,offset_s,annotator\n"
        "g1,x,1.05,3.0,me\ng2,x,2.0,4.2,me\ng3,x,0.6,1.5,me\n",
        encoding="utf-8",
    )
    fake = {"g1": (16000, 48000), "g2": (32000, 64000), "g3": (8000, 24000)}
    calls = iter(["g1", "g2", "g3"])
    monkeypatch.setattr(vad_reference, "silero_bounds", lambda audio: fake[next(calls)])
    monkeypatch.setattr(vad_reference, "silero_defaults", lambda: {"threshold": 0.5})
    monkeypatch.setattr(vad_reference, "silero_version", lambda: "test")
    out = tmp_path / "collar.json"

    rc = vle.main(
        [
            "--labels",
            str(labels),
            "--audio-dir",
            str(audio_dir),
            "--gold-template",
            str(tmp_path / "none.csv"),
            "--out",
            str(out),
            "--expect-clips",
            "3",
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["collar_s"] == pytest.approx(0.20)
    assert data["n_clips"] == 3 and data["partial"] is False
    assert data["stage1"]["collar_exceeds_clipping_limit"] is True
    assert data["stage1"]["collar_exceeds_onset_tolerance"] is True
    assert load_collar(str(out)) == (pytest.approx(0.20), False)


def test_main_with_no_labels_fails_clearly_even_when_partial_is_allowed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f = tmp_path / "labels.csv"
    f.write_text(
        "id,audio_path,onset_s,offset_s,annotator\ng1,a.wav,,,\ng2,b.wav,,,\n",
        encoding="utf-8",
    )
    out = tmp_path / "c.json"
    rc = vle.main(["--labels", str(f), "--allow-partial", "--out", str(out)])
    assert rc == 1
    assert "no labelled clips" in capsys.readouterr().err
    assert not out.exists()
