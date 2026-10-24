"""Tests for bench.vad_stage5 on small synthetic WAVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bench import vad_stage5 as s5
from bench.vad_manifest import VadItem, VadManifest
from bench.vad_runner import run_config

SR = 16000
QUIET_CELLS = [("quiet", None, p) for p in (200, 350, 500, 700)]


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


@pytest.fixture
def manifest_file(items: list[VadItem], tmp_path: Path) -> Path:
    p = tmp_path / "m.json"
    VadManifest(items=items).to_json(p)
    return p


def collar_file(tmp_path: Path, partial: bool = False) -> Path:
    p = tmp_path / "collar.json"
    p.write_text(json.dumps({"collar_s": 0.08, "partial": partial}), encoding="utf-8")
    return p


def args(tmp_path: Path, manifest: Path, collar: Path, *extra: str) -> list[str]:
    return [
        "--manifest", str(manifest),
        "--collar-file", str(collar),
        "--table", str(tmp_path / "out/table.csv"),
        "--record", str(tmp_path / "out/record.md"),
        "--raw-dir", str(tmp_path / "raw"),
        "--min-items", "2",
        "--skip-seed-check",
        "--allow-dirty",
        *extra,
    ]  # fmt: skip


def test_expected_cells_are_the_full_grid() -> None:
    assert len(s5.expected_cells()) == 10 * 4


def test_cell_row_reports_start_and_end_clipping_separately(
    items: list[VadItem],
) -> None:
    results, _ = run_config(items[:2], 0.05, 500, 0.150)
    row = s5.cell_row(0.05, 500, ("quiet", None, 200), results)
    assert set(row) == set(s5.FIELDS)
    assert row["snr_db"] == "clean" and row["items"] == 2
    assert row["clipped_start_ms_mean"] >= 0 and row["clipped_end_ms_mean"] >= 0


def test_verify_run_detects_missing_duplicate_and_small_cells() -> None:
    def row(t, o, c, s, p, n):
        return {
            "threshold": t,
            "timeout_ms": o,
            "condition": c,
            "snr_db": s,
            "pause_ms": p,
            "items": n,
        }

    cells = [("quiet", None, 200), ("quiet", None, 350)]
    good = [
        row(t, 500, "quiet", "clean", p, 20) for t in (0.05, 0.1) for p in (200, 350)
    ]
    ok = s5.verify_run(good, [0.05, 0.1], [500], cells, 20)
    assert ok["ok"] and ok["expected_rows"] == 4
    short = s5.verify_run(good[:3], [0.05, 0.1], [500], cells, 20)
    assert not short["ok"] and len(short["missing_cells"]) == 1
    small = [*good[:3], row(0.1, 500, "quiet", "clean", 350, 10)]
    assert s5.verify_run(small, [0.05, 0.1], [500], cells, 20)["cells_below_minimum"]
    dup = s5.verify_run([*good, good[0]], [0.05, 0.1], [500], cells, 20)
    assert not dup["ok"] and dup["duplicate_rows"] == 1


def test_refuses_without_a_stage3b_collar(manifest_file: Path, tmp_path: Path) -> None:
    rc = s5.main(args(tmp_path, manifest_file, tmp_path / "missing.json"))
    assert rc == 1
    assert not (tmp_path / "out/table.csv").exists()


def test_refuses_a_partial_collar(manifest_file: Path, tmp_path: Path) -> None:
    rc = s5.main(args(tmp_path, manifest_file, collar_file(tmp_path, partial=True)))
    assert rc == 1
    assert not (tmp_path / "out/table.csv").exists()


def test_full_run_writes_verified_table_and_record(
    manifest_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s5, "expected_cells", lambda: QUIET_CELLS)
    rc = s5.main(args(tmp_path, manifest_file, collar_file(tmp_path)))
    assert rc == 0
    with open(tmp_path / "out/table.csv", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == s5.FIELDS
    assert len(rows) == 25 * 4
    record = (tmp_path / "out/record.md").read_text(encoding="utf-8")
    assert "result: PASS" in record and "failed runs: 0" in record
    assert "sha256" in record and "collar: 0.08 s" in record
    assert record.count("sha256") == 3
    assert "python -m bench.vad_stage5" in record


def test_missing_cells_fail_verification_and_skip_the_table(
    manifest_file: Path, tmp_path: Path
) -> None:
    rc = s5.main(args(tmp_path, manifest_file, collar_file(tmp_path)))
    assert rc == 1
    assert not (tmp_path / "out/table.csv").exists()
    assert "result: FAIL" in (tmp_path / "out/record.md").read_text(encoding="utf-8")


def test_provisional_run_stays_in_the_raw_directory(
    manifest_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s5, "expected_cells", lambda: QUIET_CELLS)
    rc = s5.main(
        args(tmp_path, manifest_file, tmp_path / "missing.json", "--allow-provisional")
    )
    assert rc == 0
    assert (tmp_path / "raw/sweep_table.csv").exists()
    assert not (tmp_path / "out/table.csv").exists()


def test_publish_provisional_writes_a_labelled_table(
    manifest_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s5, "expected_cells", lambda: QUIET_CELLS)
    rc = s5.main(
        args(
            tmp_path,
            manifest_file,
            tmp_path / "missing.json",
            "--allow-provisional",
            "--publish-provisional",
        )
    )
    assert rc == 0
    assert (tmp_path / "out/table.csv").exists()
    record = (tmp_path / "out/record.md").read_text(encoding="utf-8")
    assert "PROVISIONAL" in record and "result: PASS" in record
