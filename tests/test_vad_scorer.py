"""Tests for bench.vad_scorer using synthetic signals and hand-built segments only.

All inputs are ground-truth and VAD segment lists in seconds; no audio files
or real data are involved.
"""

from __future__ import annotations

import pytest

from bench.vad_scorer import score_vad_item


class TestSilenceOnly:
    """No ground-truth speech, no VAD detections: all metrics zero."""

    def test_no_gt_no_vad(self) -> None:
        metrics = score_vad_item(
            gt_segments=[],
            vad_segments=[],
            total_duration_s=10.0,
            non_speech_duration_s=10.0,
            collar=0.150,
        )
        assert metrics.false_trigger_count == 0
        assert metrics.missed_onset_count == 0
        assert metrics.clipped_start_ms == 0.0
        assert metrics.clipped_end_ms == 0.0
        assert metrics.end_of_utterance_delay_ms == 0.0
        assert metrics.split_count == 0


class TestSpeechOnly:
    """VAD detects the same segment as ground truth: no false triggers, no misses."""

    def test_perfect_match(self) -> None:
        gt = [(0.5, 2.5)]
        vad = [(0.5, 2.5)]
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=5.0,
            non_speech_duration_s=3.0,
            collar=0.150,
        )
        assert metrics.false_trigger_count == 0
        assert metrics.missed_onset_count == 0
        assert metrics.clipped_start_ms == 0.0
        assert metrics.clipped_end_ms == 0.0
        assert metrics.end_of_utterance_delay_ms == 0.0
        assert metrics.split_count == 0


class TestClippedOnset:
    """VAD starts 250 ms after the ground-truth onset with a 150 ms collar:
    100 ms of speech is clipped beyond the collar tolerance."""

    def test_clipped_onset(self) -> None:
        gt = [(0.5, 2.5)]
        vad = [(0.75, 2.5)]  # 250 ms late onset, collar 150 ms -> 100 ms clipped
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=5.0,
            non_speech_duration_s=3.0,
            collar=0.150,
        )
        assert metrics.clipped_start_ms == pytest.approx(100.0, abs=1.0)
        assert metrics.missed_onset_count == 1


class TestLateOffset:
    """VAD ends 250 ms before the ground-truth offset with a 150 ms collar:
    100 ms of speech is clipped beyond the collar tolerance."""

    def test_late_offset(self) -> None:
        gt = [(0.5, 2.5)]
        vad = [(0.5, 2.25)]  # 250 ms early offset, collar 150 ms -> 100 ms clipped
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=5.0,
            non_speech_duration_s=3.0,
            collar=0.150,
        )
        assert metrics.clipped_end_ms == pytest.approx(100.0, abs=1.0)


class TestAdjacentSegments:
    """Two adjacent ground-truth segments: no overlap, no split."""

    def test_adjacent(self) -> None:
        gt = [(0.0, 1.0), (1.0, 2.0)]
        vad = [(0.0, 1.0), (1.0, 2.0)]
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=3.0,
            non_speech_duration_s=1.0,
            collar=0.150,
        )
        assert metrics.false_trigger_count == 0
        assert metrics.missed_onset_count == 0
        assert metrics.split_count == 0


class TestEmptyVadOutput:
    """No VAD detections at all: all ground-truth onsets are missed."""

    def test_empty_vad(self) -> None:
        gt = [(0.5, 2.5), (3.0, 4.0)]
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=[],
            total_duration_s=5.0,
            non_speech_duration_s=3.0,
            collar=0.150,
        )
        assert metrics.false_trigger_count == 0
        assert metrics.missed_onset_count == 2
        assert metrics.missed_onset_rate == pytest.approx(1.0)


class TestPauseSplitsUtterance:
    """A ground-truth segment split by an in-utterance pause: count as split."""

    def test_split_by_pause(self) -> None:
        gt = [(0.5, 3.5)]  # one utterance
        vad = [(0.5, 1.5), (2.3, 3.5)]  # gap 700 ms (> 500 ms timeout)
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=5.0,
            non_speech_duration_s=1.0,
            collar=0.150,
            speech_timeout_s=0.500,
        )
        assert metrics.split_count == 1


class TestCollarBoundaryDetection:
    """A VAD detection exactly at the collar boundary from a ground-truth onset
    must be counted as correct (within collar, not missed)."""

    def test_on_collar_boundary(self) -> None:
        collar = 0.150
        gt_onset = 1.0
        # Exact boundary: abs(predicted - ground_truth) == collar => correct (inclusive)
        vad_onset_exact = gt_onset + collar
        metrics = score_vad_item(
            gt_segments=[(gt_onset, 2.0)],
            vad_segments=[(vad_onset_exact, 2.0)],
            total_duration_s=5.0,
            non_speech_duration_s=3.0,
            collar=collar,
        )
        assert metrics.missed_onset_count == 0
        # Symmetric negative-side boundary
        vad_onset_neg = gt_onset - collar
        metrics = score_vad_item(
            gt_segments=[(gt_onset, 2.0)],
            vad_segments=[(vad_onset_neg, 2.0)],
            total_duration_s=5.0,
            non_speech_duration_s=3.0,
            collar=collar,
        )
        assert metrics.missed_onset_count == 0


class TestFalseTriggerByStart:
    """False trigger defined by VAD start occurring in non-speech.
    Start inside GT segment or within collar of GT onset = not false.
    Start in pure silence = false trigger."""

    def test_vad_start_inside_non_speech(self) -> None:
        gt = [(2.0, 4.0)]
        vad = [(5.0, 6.0)]  # start clearly in non-speech
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=7.0,
            non_speech_duration_s=3.0,
            collar=0.150,
        )
        assert metrics.false_trigger_count == 1
        assert metrics.false_trigger_rate_per_min == pytest.approx(20.0)

    def test_vad_start_within_collar_of_gt_onset_not_false(self) -> None:
        gt = [(2.0, 4.0)]
        vad = [(1.95, 3.5)]  # start 50 ms before GT onset, within 150 ms collar
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=7.0,
            non_speech_duration_s=3.0,
            collar=0.150,
        )
        assert metrics.false_trigger_count == 0

    def test_vad_segment_overlaps_speech_but_starts_in_non_speech(self) -> None:
        # VAD starts at 1.5 (before GT 2.0, outside collar of 2.0 -> non-speech start)
        # but overlaps GT because VAD extends to 3.5.
        # Per start-based definition: start is in non-speech => false trigger.
        gt = [(2.0, 4.0)]
        vad = [(1.5, 3.5)]
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=7.0,
            non_speech_duration_s=3.0,
            collar=0.150,
        )
        assert metrics.false_trigger_count == 1


class TestFalseTrigger:
    """A VAD segment with no overlapping ground-truth segment is a false trigger."""

    def test_false_trigger_count(self) -> None:
        gt = [(1.0, 2.0)]
        vad = [(3.0, 4.0)]  # no overlap with GT
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=5.0,
            non_speech_duration_s=3.0,
            collar=0.150,
        )
        assert metrics.false_trigger_count == 1
        assert metrics.false_trigger_rate_per_min == pytest.approx(20.0)


class TestEndToEndOfDelay:
    """VAD ends after the ground-truth offset: end-of-utterance delay measured."""

    def test_eou_delay(self) -> None:
        gt = [(0.5, 2.5)]
        vad = [(0.5, 2.8)]  # 300 ms late end
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=5.0,
            non_speech_duration_s=3.0,
            collar=0.150,
        )
        assert metrics.end_of_utterance_delay_ms == pytest.approx(300.0, abs=1.0)


class TestMissingOnset:
    """A VAD segment that starts far from the ground-truth onset misses the onset."""

    def test_missed_onset_far(self) -> None:
        gt = [(1.0, 2.0)]
        vad = [(1.5, 2.0)]  # onset 500 ms after GT onset, beyond collar
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=5.0,
            non_speech_duration_s=3.0,
            collar=0.150,
        )
        assert metrics.missed_onset_count == 1


class TestSplitWithPause:
    """A pause within a ground-truth utterance that creates two VAD segments
    counts as one split utterance."""

    def test_split_by_in_utterance_pause(self) -> None:
        gt = [(0.0, 4.0)]
        vad = [(0.0, 1.5), (2.3, 4.0)]  # gap 700 ms (> 500 ms timeout)
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=5.0,
            non_speech_duration_s=1.0,
            collar=0.150,
            speech_timeout_s=0.500,
        )
        assert metrics.split_count == 1
        assert metrics.false_trigger_count == 0


class TestSplitByTimeout:
    """Split utterance evaluated with speech-timeout semantics (500 ms)."""

    def test_gap_200_ms_not_split(self) -> None:
        gt = [(0.0, 4.0)]
        vad = [(0.0, 1.8), (2.0, 4.0)]  # gap 200 ms
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=5.0,
            non_speech_duration_s=1.0,
            collar=0.150,
            speech_timeout_s=0.500,
        )
        assert metrics.split_count == 0

    def test_gap_350_ms_not_split(self) -> None:
        gt = [(0.0, 4.0)]
        vad = [(0.0, 1.8), (2.15, 4.0)]  # gap 350 ms
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=5.0,
            non_speech_duration_s=1.0,
            collar=0.150,
            speech_timeout_s=0.500,
        )
        assert metrics.split_count == 0

    def test_gap_500_ms_not_split(self) -> None:
        gt = [(0.0, 4.0)]
        vad = [(0.0, 1.5), (2.0, 4.0)]  # gap 500 ms
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=5.0,
            non_speech_duration_s=1.0,
            collar=0.150,
            speech_timeout_s=0.500,
        )
        assert metrics.split_count == 0

    def test_gap_700_ms_split(self) -> None:
        gt = [(0.0, 4.0)]
        vad = [(0.0, 1.8), (2.5, 4.0)]  # gap 700 ms (> 500 ms timeout)
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=5.0,
            non_speech_duration_s=1.0,
            collar=0.150,
            speech_timeout_s=0.500,
        )
        assert metrics.split_count == 1


class TestEOUDelayNegative:
    """End-of-utterance delay preserves negative values when VAD ends before true end."""

    def test_vad_ends_before_true_end(self) -> None:
        gt = [(0.5, 2.5)]
        vad = [(0.5, 2.3)]  # VAD ends 200 ms before GT end
        metrics = score_vad_item(
            gt_segments=gt,
            vad_segments=vad,
            total_duration_s=5.0,
            non_speech_duration_s=3.0,
            collar=0.150,
        )
        assert metrics.end_of_utterance_delay_ms == pytest.approx(-200.0, abs=1.0)
