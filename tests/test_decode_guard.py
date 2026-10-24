"""Tests for the per-utterance decode-length guard (no model weights).

The guard exists because CTranslate2's greedy decode looped to a flat 256-token
cap on 12 of 50 VI->EN sentences, and those items set the device latency p95.
These tests pin the properties that make it a guard rather than a clipping bug:
it scales with the input, never truncates a legitimate output, and always has a
hard ceiling.
"""

from __future__ import annotations

import inspect

import pytest

from bench.candidates.m2m_mt import M2M100MTCandidate
from bench.candidates.opusmt_mt import OpusMTMTCandidate
from bench.decode import CEILING, FLOOR, RATIO, max_decoding_length


def test_short_input_hits_the_floor():
    assert max_decoding_length(1) == FLOOR
    assert max_decoding_length(10) == FLOOR
    # 1.5 x 20 = 30, still under the floor
    assert max_decoding_length(20) == FLOOR


def test_long_input_hits_the_ceiling():
    # Above ~171 pieces the 1.5x rule would exceed the ceiling, so it clamps.
    assert max_decoding_length(200) == CEILING
    assert max_decoding_length(5000) == CEILING


def test_middle_input_scales_with_the_ratio():
    assert max_decoding_length(60) == pytest.approx(90)
    assert max_decoding_length(100) == pytest.approx(150)


def test_cap_is_monotonic():
    caps = [max_decoding_length(n) for n in range(1, 400, 7)]
    assert caps == sorted(caps)


def test_cap_never_clips_an_observed_reference_length():
    """1.5x sits above the longest target/source ratio measured on 347 pairs.

    p99 is 1.17 and the maximum observed is 1.28, so the guard must not bind for
    any input longer than the floor — otherwise it would truncate real
    translations instead of only runaway repetition.
    """
    observed_max_ratio = 1.28
    assert observed_max_ratio < RATIO
    for n in (32, 48, 64, 96, 128, 168):
        assert max_decoding_length(n) >= observed_max_ratio * n


def test_zero_and_negative_input_do_not_crash():
    assert max_decoding_length(0) == FLOOR
    assert max_decoding_length(-5) == FLOOR


def test_candidates_no_longer_pin_a_flat_cap():
    """The cap must come from the item, not from the constructor config."""
    for cls in (OpusMTMTCandidate, M2M100MTCandidate):
        source = inspect.getsource(cls)
        assert "max_decoding_length=256" not in source
        assert "max_decoding_length=max_decoding_length(len(tokens))" in source
