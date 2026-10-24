"""Tests for bench.vad_borderline."""

from __future__ import annotations

import pytest

from bench import vad_borderline as vb


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("missed_onset_rate", 0.0, "pass"),
        ("missed_onset_rate", 0.03, "pass"),
        ("missed_onset_rate", 0.045, "borderline"),
        ("missed_onset_rate", 0.05, "borderline"),
        ("missed_onset_rate", 0.058, "borderline"),
        ("missed_onset_rate", 0.07, "fail"),
        ("false_trigger_rate_per_min", 0.0, "pass"),
        ("false_trigger_rate_per_min", 0.05, "pass"),
        ("false_trigger_rate_per_min", 0.09, "borderline"),
        ("false_trigger_rate_per_min", 0.11, "borderline"),
        ("false_trigger_rate_per_min", 0.2, "fail"),
        ("clipped_ms_per_utterance", 20.0, "pass"),
        ("clipped_ms_per_utterance", 41.0, "borderline"),
        ("clipped_ms_per_utterance", 59.0, "borderline"),
        ("clipped_ms_per_utterance", 61.0, "fail"),
    ],
)
def test_assess(name: str, value: float, expected: str) -> None:
    assert vb.assess(name, value) == expected


def test_assess_without_value_is_undetermined() -> None:
    assert vb.assess("missed_onset_rate", None) == "undetermined"


def test_borderline_is_never_a_pass() -> None:
    for name in vb.CRITERIA:
        limit = vb.CRITERIA[name].limit
        assert vb.assess(name, limit) == "borderline"


def test_margins() -> None:
    assert vb.margin(vb.Criterion(0.5, "proportion")) == pytest.approx(0.05)
    assert vb.margin(vb.Criterion(0.1, "rate")) == pytest.approx(0.02)
    assert vb.margin(vb.Criterion(50.0, "ms")) == pytest.approx(10.0)


def test_needs_larger_sample() -> None:
    assert vb.needs_larger_sample(["pass", "borderline", "fail"])
    assert not vb.needs_larger_sample(["pass", "fail", "undetermined"])
    assert vb.ITEMS_PER_CELL_WHEN_BORDERLINE == 2 * vb.ITEMS_PER_CELL


def test_false_trigger_upper_bound_rule_of_three() -> None:
    assert vb.false_trigger_upper95(0, 2.0) == pytest.approx(1.5)
    assert vb.false_trigger_upper95(1, 2.0) is None
    assert vb.false_trigger_upper95(0, 0.0) is None
