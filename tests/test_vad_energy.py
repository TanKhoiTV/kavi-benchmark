"""Tests for bench.vad_energy using synthetic signals only."""

from __future__ import annotations

import numpy as np
import pytest

from bench.vad_energy import EnergyVad

SR = 16000


def silence(ms: int) -> np.ndarray:
    return np.zeros(ms * 16, dtype=np.float32)


def tone(ms: int, amp: float = 0.5, freq: float = 440.0) -> np.ndarray:
    t = np.arange(ms * 16) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def cat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts)


def test_silence_has_no_segments() -> None:
    result = EnergyVad().detect(silence(2000))
    assert result.segments == []
    assert result.eos_s == []


def test_single_burst() -> None:
    audio = cat(silence(500), tone(300), silence(1000))
    result = EnergyVad(0.05, 500).detect(audio)
    assert result.segments == [pytest.approx((0.5, 0.8), abs=1e-9)]
    assert result.eos_s == [pytest.approx(0.8 + 0.51, abs=1e-9)]


def test_pause_shorter_than_timeout_does_not_split() -> None:
    audio = cat(silence(500), tone(300), silence(300), tone(300), silence(1000))
    result = EnergyVad(0.05, 500).detect(audio)
    assert result.segments == [pytest.approx((0.5, 1.4), abs=1e-9)]


def test_pause_equal_to_timeout_does_not_split() -> None:
    audio = cat(silence(500), tone(300), silence(500), tone(300), silence(1000))
    result = EnergyVad(0.05, 500).detect(audio)
    assert len(result.segments) == 1


def test_pause_one_frame_over_timeout_splits() -> None:
    audio = cat(silence(500), tone(300), silence(510), tone(300), silence(1000))
    result = EnergyVad(0.05, 500).detect(audio)
    assert len(result.segments) == 2


def test_pause_longer_than_timeout_splits() -> None:
    audio = cat(silence(500), tone(300), silence(700), tone(300), silence(1000))
    result = EnergyVad(0.05, 500).detect(audio)
    assert result.segments == [
        pytest.approx((0.5, 0.8), abs=1e-9),
        pytest.approx((1.5, 1.8), abs=1e-9),
    ]


@pytest.mark.parametrize(
    ("timeout_ms", "expected_segments"), [(200, 2), (290, 2), (300, 1), (350, 1)]
)
def test_timeout_is_a_parameter(timeout_ms: int, expected_segments: int) -> None:
    audio = cat(silence(500), tone(300), silence(300), tone(300), silence(1000))
    assert len(EnergyVad(0.05, timeout_ms).detect(audio).segments) == expected_segments


@pytest.mark.parametrize(
    ("threshold", "expected_segments"), [(0.02, 1), (0.04, 1), (0.05, 0), (0.2, 0)]
)
def test_threshold_is_a_parameter(threshold: float, expected_segments: int) -> None:
    audio = cat(silence(300), tone(300, amp=0.045), silence(1000))
    assert len(EnergyVad(threshold, 500).detect(audio).segments) == expected_segments


def test_threshold_comparison_is_inclusive() -> None:
    audio = cat(silence(100), np.full(1600, 0.25, dtype=np.float32), silence(1000))
    assert len(EnergyVad(0.25, 500).detect(audio).segments) == 1


def test_audio_ending_inside_speech_closes_without_eos() -> None:
    audio = cat(silence(200), tone(300))
    result = EnergyVad(0.05, 500).detect(audio)
    assert result.segments == [pytest.approx((0.2, 0.5), abs=1e-9)]
    assert result.eos_s == [None]


def test_audio_ending_before_timeout_elapses_has_no_eos() -> None:
    audio = cat(silence(200), tone(300), silence(400))
    assert EnergyVad(0.05, 500).detect(audio).eos_s == [None]


def test_length_not_a_multiple_of_frame_keeps_offset_inside_audio() -> None:
    audio = cat(silence(200), tone(300))[:-37]
    start, end = EnergyVad(0.05, 500).detect(audio).segments[0]
    assert start == pytest.approx(0.2, abs=1e-9)
    assert end <= len(audio) / SR + 1e-9


@pytest.mark.parametrize("timeout_ms", [0, -10, 345])
def test_invalid_timeout_rejected(timeout_ms: int) -> None:
    with pytest.raises(ValueError):
        EnergyVad(0.05, timeout_ms)


def test_invalid_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        EnergyVad(0.0, 500)


@pytest.mark.parametrize("threshold", [0.01, 0.05, 0.2])
@pytest.mark.parametrize("timeout_ms", [200, 350, 1000])
def test_detect_energy_matches_detect(threshold: float, timeout_ms: int) -> None:
    rng = np.random.default_rng(3)
    audio = cat(
        silence(300),
        (0.4 * rng.standard_normal(4000)).astype(np.float32),
        silence(450),
        tone(300, amp=0.08),
        silence(900),
    )[:-37]
    vad = EnergyVad(threshold, timeout_ms)
    direct = vad.detect(audio)
    via_energy = vad.detect_energy(vad.frame_energy(audio), len(audio))
    assert via_energy == direct
