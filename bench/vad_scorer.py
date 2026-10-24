"""Off-device VAD scorer: false triggers, missed onsets, clipped speech, end-of-utterance delay, split utterances.

Scoring compares ground-truth speech segments against VAD output segments using
a symmetric collar tolerance. All inputs are segment lists of (onset_s, offset_s)
tuples; no audio and no VAD inference are involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VadMetrics:
    """Container for all VAD scoring metrics on a single utterance."""

    false_trigger_count: int = 0
    false_trigger_rate_per_min: float = 0.0
    missed_onset_count: int = 0
    missed_onset_rate: float = 0.0
    clipped_start_ms: float = 0.0
    clipped_end_ms: float = 0.0
    end_of_utterance_delay_ms: float = 0.0
    split_count: int = 0

    errors: list[str] = field(default_factory=list)


def _is_in_collar(vad_boundary: float, gt_boundary: float, collar: float) -> bool:
    """Return True if the VAD boundary falls within +/- collar seconds of the ground-truth boundary (inclusive, with float tolerance)."""
    return abs(vad_boundary - gt_boundary) <= collar + 1e-9


def _segments_overlap(
    seg_a_start: float, seg_a_end: float, seg_b_start: float, seg_b_end: float
) -> bool:
    """Return True if [seg_a_start, seg_a_end) overlaps [seg_b_start, seg_b_end)."""
    return seg_a_start < seg_b_end and seg_b_start < seg_a_end


def _is_false_trigger(
    vad_seg, gt_segments: list[tuple[float, float]], collar: float
) -> bool:
    """Return True when VAD start occurs in non-speech (no GT onset within collar, not inside GT segment)."""
    vs, _ve = vad_seg
    for gs, ge in gt_segments:
        if abs(vs - gs) <= collar:
            return False
        if gs <= vs <= ge:
            return False
    return True


def _is_missed_onset(
    gt_onset: float, vad_segments: list[tuple[float, float]], collar: float
) -> bool:
    """Return True if no VAD segment onset falls within +/- collar seconds of the ground-truth onset."""
    return all(not _is_in_collar(vs, gt_onset, collar) for vs, _ve in vad_segments)


def _clip_at_start(vad_seg, gt_seg, collar: float) -> float:
    """Return the milliseconds of the ground-truth segment lost before the VAD segment onset (after collar), or 0.0 if none."""
    vs, _ve = vad_seg
    gs, _ge = gt_seg
    delay = (vs - gs) - collar
    return max(0.0, delay * 1000.0)


def _clip_at_end(vad_seg, gt_seg, collar: float) -> float:
    """Return the milliseconds of the ground-truth segment lost after the VAD segment offset (after collar), or 0.0 if none."""
    _vs, ve = vad_seg
    _gs, ge = gt_seg
    delay = ge - ve - collar
    return max(0.0, delay * 1000.0)


def _is_split_utterance(
    gt_seg,
    vad_segments: list[tuple[float, float]],
    collar: float,
    speech_timeout_s: float = 0.500,
) -> bool:
    """Return True when a GT utterance is broken into multiple VAD segments by a real in-utterance gap (> speech_timeout_s)."""
    gs, ge = gt_seg
    overlaps = [
        (vs, ve) for vs, ve in vad_segments if _segments_overlap(vs, ve, gs, ge)
    ]
    if len(overlaps) < 2:
        return False
    overlaps_sorted = sorted(overlaps, key=lambda s: s[0])
    for i in range(1, len(overlaps_sorted)):
        prev_end = overlaps_sorted[i - 1][1]
        curr_start = overlaps_sorted[i][0]
        gap = curr_start - prev_end
        if gap > 0 and gap > speech_timeout_s:
            return True
    return False


def score_vad_item(
    gt_segments: list[tuple[float, float]],
    vad_segments: list[tuple[float, float]],
    total_duration_s: float,
    non_speech_duration_s: float,
    collar: float = 0.150,
    speech_timeout_s: float = 0.500,
) -> VadMetrics:
    """Score a single utterance's VAD output against ground-truth speech segments.

    Parameters
    ----------
    gt_segments:
        Ground-truth speech segments as (onset_s, offset_s) tuples.
    vad_segments:
        VAD-detected speech segments as (onset_s, offset_s) tuples.
    total_duration_s:
        Total audio duration in seconds (for false-trigger rate denominator).
    non_speech_duration_s:
        Duration of non-speech audio in seconds (for false-trigger rate).
    collar:
        Symmetric tolerance in seconds applied to all boundary comparisons.
    speech_timeout_s:
        Speech timeout in seconds (default 0.500 s, from .kavi.yaml timing); used
        only for split-utterance evaluation, not for collar-based boundary checks.

    Returns
    -------
    VadMetrics:
        All VAD metrics for this utterance.
    """
    metrics = VadMetrics()

    if non_speech_duration_s <= 0:
        metrics.errors.append(
            "non_speech_duration_s is zero; false-trigger rate undefined"
        )
        return metrics

    if total_duration_s <= 0:
        metrics.errors.append(
            "total_duration_s is zero; cannot compute duration-based metrics"
        )
        return metrics

    gt_segments_sorted = sorted(gt_segments, key=lambda s: s[0])
    vad_segments_sorted = sorted(vad_segments, key=lambda s: s[0])

    false_triggers = [
        seg
        for seg in vad_segments_sorted
        if _is_false_trigger(seg, gt_segments_sorted, collar)
    ]
    metrics.false_trigger_count = len(false_triggers)
    metrics.false_trigger_rate_per_min = metrics.false_trigger_count / (
        non_speech_duration_s / 60.0
    )

    missed_onsets = [
        (gs, ge)
        for gs, ge in gt_segments_sorted
        if _is_missed_onset(gs, vad_segments_sorted, collar)
    ]
    metrics.missed_onset_count = len(missed_onsets)
    total_gt = len(gt_segments_sorted)
    metrics.missed_onset_rate = (
        metrics.missed_onset_count / total_gt if total_gt > 0 else 0.0
    )

    for gs, ge in gt_segments_sorted:
        overlapping = [
            (vs, ve)
            for vs, ve in vad_segments_sorted
            if _segments_overlap(vs, ve, gs, ge)
            and not _is_false_trigger((vs, ve), gt_segments_sorted, collar)
        ]
        if overlapping:
            best_start = min(overlapping, key=lambda s: s[0])
            best_end = max(overlapping, key=lambda s: s[1])
            metrics.clipped_start_ms += _clip_at_start(best_start, (gs, ge), collar)
            metrics.clipped_end_ms += _clip_at_end(best_end, (gs, ge), collar)

    best_delay_s_for_utterance = None
    for gs, ge in gt_segments_sorted:
        best_delay_s = None
        for vs, ve in vad_segments_sorted:
            if _segments_overlap(vs, ve, gs, ge):
                delay = ve - ge
                if best_delay_s is None or delay > best_delay_s:
                    best_delay_s = delay
        if best_delay_s is not None and (
            best_delay_s_for_utterance is None
            or best_delay_s > best_delay_s_for_utterance
        ):
            best_delay_s_for_utterance = best_delay_s
    if best_delay_s_for_utterance is not None:
        metrics.end_of_utterance_delay_ms = best_delay_s_for_utterance * 1000.0

    for gs, ge in gt_segments_sorted:
        if _is_split_utterance((gs, ge), vad_segments_sorted, collar, speech_timeout_s):
            metrics.split_count += 1

    return metrics
