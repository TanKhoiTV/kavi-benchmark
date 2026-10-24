"""Tests for bench.vad_runner using small synthetic WAVs written to tmp_path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bench.vad_manifest import VadItem, VadManifest
from bench.vad_runner import (
    PROVISIONAL_COLLAR_S,
    aggregate,
    estimate_sweep,
    load_collar,
    main,
    merged_gt_s,
    run_config,
    select_items,
)

SR = 16000
LEN_A = 16000
LEN_B = 16000


@pytest.fixture
def wavs(tmp_path: Path) -> dict[str, str]:
    rng = np.random.default_rng(0)
    paths = {}
    for name in ("a", "b"):
        p = tmp_path / f"{name}.wav"
        sf.write(str(p), (0.3 * rng.standard_normal(LEN_A)).astype("float32"), SR)
        paths[name] = str(p)
    return paths


def make_item(wavs: dict[str, str], pause_ms: int, item_id: str = "x") -> VadItem:
    a0 = 16000
    b0 = a0 + LEN_A + pause_ms * 16
    return VadItem(
        id=item_id,
        language="vi",
        condition="quiet",
        clean_audio_path=wavs["a"],
        noise_type="clean",
        speech_b_path=wavs["b"],
        pause_ms=pause_ms,
        ground_truth_segments=[(a0, a0 + LEN_A), (b0, b0 + LEN_B)],
    )


def test_load_collar_missing_file_is_provisional(tmp_path: Path) -> None:
    assert load_collar(str(tmp_path / "none.json")) == (PROVISIONAL_COLLAR_S, True)


def test_load_collar_reads_file(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"collar_s": 0.2}), encoding="utf-8")
    assert load_collar(str(p)) == (0.2, False)


@pytest.mark.parametrize("bad", [{}, {"collar_s": 0}, {"collar_s": "x"}])
def test_load_collar_rejects_bad_file(tmp_path: Path, bad: dict) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_collar(str(p))


def test_merged_gt_spans_both_segments(wavs: dict[str, str]) -> None:
    item = make_item(wavs, 200)
    start, end = merged_gt_s(item)[0]
    assert start == 1.0
    assert end == item.ground_truth_segments[1][1] / SR


def test_select_items_takes_n_per_cell_and_language(wavs: dict[str, str]) -> None:
    items = []
    for lang in ("vi", "en"):
        for k in range(3):
            it = make_item(wavs, 200, item_id=f"{lang}-{k}")
            it.language = lang
            items.append(it)
    picked = select_items(items, 1)
    assert [i.id for i in picked] == ["vi-0", "en-0"]
    assert len(select_items(items, None)) == 6


def test_run_config_pause_within_and_beyond_timeout(wavs: dict[str, str]) -> None:
    items = [make_item(wavs, 200, "p200"), make_item(wavs, 700, "p700")]
    results, timing = run_config(items, 0.05, 500, 0.150)
    short, long_ = results
    assert short.n_vad_segments == 1
    assert short.metrics.split_count == 0
    assert short.violation is False
    assert long_.n_vad_segments == 2
    assert long_.metrics.split_count == 1
    assert long_.violation is False
    assert short.junction_split is False
    assert long_.junction_split is True
    assert short.junction_violation is False
    assert long_.junction_violation is False
    for r in results:
        assert r.metrics.false_trigger_count == 0
        assert r.metrics.missed_onset_count == 0
        assert r.eos_delay_ms == pytest.approx(510.0, abs=1.0)
    assert timing["wall_s"] > 0


def test_run_config_all_below_threshold_misses_onset(wavs: dict[str, str]) -> None:
    results, _ = run_config([make_item(wavs, 200)], 5.0, 500, 0.150)
    assert results[0].n_vad_segments == 0
    assert results[0].metrics.missed_onset_count == 1
    assert results[0].eos_delay_ms is None


def test_aggregate_groups_by_condition_and_snr(wavs: dict[str, str]) -> None:
    items = [make_item(wavs, 200, "p200"), make_item(wavs, 700, "p700")]
    results, _ = run_config(items, 0.05, 500, 0.150)
    agg = aggregate(results)
    assert len(agg["cells"]) == 1
    cell = agg["cells"][0]
    assert (cell["condition"], cell["snr_db"], cell["n_items"]) == ("quiet", None, 2)
    assert cell["split_rate_by_pause"] == {"200": 0.0, "700": 1.0}
    assert cell["pause_violations"] == 0
    assert cell["junction_split_rate_by_pause"] == {"200": 0.0, "700": 1.0}
    assert cell["junction_violations"] == 0
    assert cell["utterance_split_rate"] == 0.5
    assert cell["missed_onset_rate"] == 0.0
    assert agg["overall"]["n_items"] == 2


def test_estimate_sweep_scales_with_grid() -> None:
    timing = {"build_s": 2.0, "vad_s": 0.5, "score_s": 0.5}
    est = estimate_sweep(timing, n_run=10, n_full=100, n_configs=25)
    assert est["rebuild_audio_every_config_s"] == pytest.approx(25 * 100 * 0.3)
    assert est["build_audio_once_s"] == pytest.approx(100 * 0.2 + 25 * 100 * 0.1)


def test_main_writes_json_and_marks_collar_provisional(
    wavs: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "m.json"
    VadManifest(items=[make_item(wavs, 200), make_item(wavs, 700, "y")]).to_json(
        manifest
    )
    out = tmp_path / "out"
    rc = main(
        [
            "--manifest",
            str(manifest),
            "--collar-file",
            str(tmp_path / "missing.json"),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert "PROVISIONAL" in capsys.readouterr().out
    data = json.loads((out / "vad_thr0.05_to500.json").read_text(encoding="utf-8"))
    assert data["collar_provisional"] is True
    assert data["config"]["n_items"] == 2
    assert data["overall"]["n_items"] == 2


def make_trimmed_item(
    wavs: dict[str, str], pause_ms: int, gt_ok: bool = True
) -> VadItem:
    item = make_item(wavs, pause_ms, "trim")
    item.speech_a_bounds = (2000, 12000)
    item.speech_b_bounds = (1000, 15000)
    if gt_ok:
        a_end = 16000 + 10000
        b_start = a_end + pause_ms * 16
        item.ground_truth_segments = [(16000, a_end), (b_start, b_start + 14000)]
    return item


def test_run_config_uses_source_bounds(wavs: dict[str, str]) -> None:
    results, _ = run_config([make_trimmed_item(wavs, 200)], 0.05, 500, 0.150)
    assert results[0].metrics.missed_onset_count == 0
    assert results[0].n_vad_segments == 1


def test_run_config_rejects_stale_manifest_gt(wavs: dict[str, str]) -> None:
    with pytest.raises(RuntimeError, match="stale manifest"):
        run_config([make_trimmed_item(wavs, 200, gt_ok=False)], 0.05, 500, 0.150)


def test_inner_gap_is_not_a_junction_split(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    a = (0.3 * rng.standard_normal(LEN_A)).astype("float32")
    a[5000 : 5000 + 9600] = 0.0
    b = (0.3 * rng.standard_normal(LEN_B)).astype("float32")
    paths = {}
    for name, x in (("a", a), ("b", b)):
        p = tmp_path / f"{name}.wav"
        sf.write(str(p), x, SR)
        paths[name] = str(p)
    item = make_item(paths, 200, "hole")
    result = run_config([item], 0.05, 500, 0.150)[0][0]
    assert result.metrics.split_count == 1
    assert result.violation is True
    assert result.junction_split is False
    assert result.junction_violation is False
