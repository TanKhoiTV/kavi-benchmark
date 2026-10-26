"""Tests for the pause-distribution measurement, without torch or Silero."""

from __future__ import annotations

from bench.pause_distribution import (
    FRAME_MS,
    SENTENCE_PAUSE_MS,
    pauses_ms,
    percentile,
    summarise,
    table,
)

SR = 16000


def stamp(start_ms: float, end_ms: float) -> dict:
    return {"start": int(start_ms * SR / 1000), "end": int(end_ms * SR / 1000)}


def test_percentile_is_nearest_rank_without_interpolation() -> None:
    # rank = ceil(p/100 * n); no averaging, so every returned value is observed.
    assert percentile([100.0, 200.0], 50) == 100.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 99) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 1) == 1.0


def test_percentile_of_nothing_is_zero() -> None:
    assert percentile([], 95) == 0.0


def test_pauses_are_gaps_between_segments_only() -> None:
    # Leading and trailing silence are not pauses: the first segment starts at 200 ms
    # and the last ends at 900 ms, but only the gap between them counts.
    stamps = [stamp(200, 500), stamp(700, 900)]
    assert pauses_ms(stamps) == [200.0]


def test_single_segment_has_no_pauses() -> None:
    assert pauses_ms([stamp(100, 900)]) == []


def test_multiple_gaps_are_all_reported() -> None:
    stamps = [stamp(0, 100), stamp(150, 300), stamp(900, 1000)]
    assert pauses_ms(stamps) == [50.0, 600.0]


def test_summarise_counts_pauses_and_percentiles() -> None:
    pauses = [100.0, 200.0, 300.0, 400.0, 500.0]
    stats = summarise(pauses, clips=4)
    assert stats["pauses"] == 5
    assert stats["clips"] == 4
    assert stats["max_ms"] == 500.0
    assert stats["p50_ms"] == 300.0
    # "reaching" is >=, so a pause exactly at the timeout counts as reaching it.
    assert stats["reaching"]["500"] == 1
    assert stats["reaching"]["450"] == 1  # only 500 clears 450
    assert stats["reaching"]["200"] == 4


def test_summarise_separates_inter_sentence_gaps() -> None:
    stats = summarise([200.0, SENTENCE_PAUSE_MS, SENTENCE_PAUSE_MS + 100.0], clips=2)
    assert stats["over_word"] == 2


def test_summarise_of_no_pauses_is_zero_not_an_error() -> None:
    stats = summarise([], clips=3)
    assert stats["pauses"] == 0
    assert stats["max_ms"] == 0.0
    assert stats["p50_ms"] == 0.0


def test_table_reports_both_sets_and_every_candidate() -> None:
    stats = summarise([150.0, 400.0], clips=1)
    stats["inter_word"] = summarise([150.0], clips=1)
    stats["truncated_clips"] = {str(t): 0 for t in (200, 250, 300, 350, 400, 450, 500)}
    stats["threshold"] = 0.5
    stats["clips"] = 1
    stats["mean_clip_s"] = 3.0
    rendered = table("VI", stats)
    assert "max" in rendered and "400.0" in rendered
    for timeout in (200, 250, 300, 350, 400, 450, 500):
        assert f"| {timeout} ms |" in rendered


def test_frame_resolution_is_documented_as_32ms() -> None:
    # The write-up claims 32 ms resolution; a timeout must clear the max by one frame.
    assert FRAME_MS == 32.0
