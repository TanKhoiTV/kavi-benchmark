"""Borderline rule for the hard VAD criteria.

A metric inside the margin around its criterion is BORDERLINE: it is not counted as
a pass (the stricter reading applies) and the cell needs 40 items instead of 20.
The criterion itself is never relaxed.

Margins: criteria of kind "proportion" use +/-5 percentage points; criteria of kind
"rate" and "ms" use 20% of the criterion.

ASSUMPTIONS (the Stage 1 text is not in the repository; confirm before relying on it):
- missed_onset_rate (<= 5%) is a "rate", so its margin is 20% of 5% = 1 point. Read as
  +/-5 percentage points, every value from 0% to 10% would be borderline and the
  criterion could never be passed.
- clipped_ms_per_utterance (<= 50 ms) is treated like a rate, margin 10 ms.
"""

from __future__ import annotations

from dataclasses import dataclass

PROPORTION_MARGIN = 0.05
RELATIVE_MARGIN = 0.20
ITEMS_PER_CELL = 20
ITEMS_PER_CELL_WHEN_BORDERLINE = 40


@dataclass(frozen=True)
class Criterion:
    limit: float
    kind: str


CRITERIA: dict[str, Criterion] = {
    "missed_onset_rate": Criterion(0.05, "rate"),
    "false_trigger_rate_per_min": Criterion(0.1, "rate"),
    "clipped_ms_per_utterance": Criterion(50.0, "ms"),
}


def margin(criterion: Criterion) -> float:
    if criterion.kind == "proportion":
        return PROPORTION_MARGIN
    return RELATIVE_MARGIN * criterion.limit


def assess(name: str, value: float | None) -> str:
    """Return pass, fail, borderline, or undetermined (no value)."""
    if value is None:
        return "undetermined"
    criterion = CRITERIA[name]
    if abs(value - criterion.limit) <= margin(criterion):
        return "borderline"
    return "pass" if value < criterion.limit else "fail"


def needs_larger_sample(statuses: list[str]) -> bool:
    return "borderline" in statuses


def false_trigger_upper95(events: int, non_speech_min: float) -> float | None:
    """95% upper bound on the false-trigger rate per minute when no event was seen
    (rule of three); None when events were observed or there is no non-speech time."""
    if events != 0 or non_speech_min <= 0:
        return None
    return 3.0 / non_speech_min
