"""Model-free tests for the formal beam sweep (no Opus/M2M weights).

The sweep exists to replace a set of bad measurements with defensible ones, so
these tests pin the properties that make the output citable: warmup is discarded,
every repeat is kept, the beam width is recorded, KV cache is derived from
geometry rather than RSS deltas, and both BLEU definitions are reported.
"""

from __future__ import annotations

import statistics

import pytest

from bench.adapters import Candidate
from bench.beam_sweep import (
    bleu_scores,
    kv_cache_mb,
    monotonic_breaks,
    mt_items,
    per_item_medians,
    summarise,
    sweep,
    timed_rows,
)
from bench.schema import EvalItem, RunManifest


def make_item(index: int) -> EvalItem:
    return EvalItem(
        id=f"mt-{index:03d}",
        stage="MT",
        language="vi",
        direction="vi->en",
        input_text=f"câu {index}",
        reference_text=f"the sentence number {index} here",
    )


class FakeCandidate(Candidate):
    """Records every call; ``latency_s`` is derived from the configured beam."""

    stage = "MT"
    id = "fake-mt"

    def __init__(self) -> None:
        self.calls: list[int] = []
        self._beam = 1

    def configure(self, config: dict) -> None:
        self._beam = int(config["beam_size"])

    def _infer(self, item: EvalItem):  # type: ignore[override]
        self.calls.append(self._beam)
        return item.reference_text, None


def test_sweep_discards_warmup_and_keeps_every_repeat():
    items = [make_item(i) for i in range(3)]
    candidate = FakeCandidate()

    rows = sweep(items, candidate, [1, 4], repeats=2, warmup=1)

    # 2 beams x (1 warmup + 2 repeats) x 3 items reach the candidate
    assert len(candidate.calls) == 2 * 3 * 3
    # ...but only the 2 repeats per item are reported
    assert len(rows) == 2 * 2 * 3


def test_sweep_reconfigures_in_place_rather_than_reloading():
    items = [make_item(0)]
    candidate = FakeCandidate()

    rows = sweep(items, candidate, [1, 2, 4], repeats=1, warmup=0)

    assert [row["beam"] for row in rows] == [1, 2, 4]
    # One instance served all three beams: the constructor ran once, so the
    # candidate never reloads weights per setting (issue #90, problem 4).
    assert candidate.calls == [1, 2, 4]


def test_rows_record_the_beam_width_that_produced_them():
    items = [make_item(0)]
    rows = sweep(items, FakeCandidate(), [1, 8], repeats=1, warmup=0)

    assert {row["beam"] for row in rows} == {1, 8}


def test_timed_rows_drops_errors_and_zero_latency():
    rows = [
        {"latency_s": 1.0, "error": None},
        {"latency_s": 0.0, "error": None},
        {"latency_s": 2.0, "error": "boom"},
    ]

    assert len(timed_rows(rows)) == 1


def test_per_item_medians_uses_the_median_not_the_mean():
    items = [make_item(0)]
    rows = [
        {
            "item_id": "mt-000",
            "beam": 1,
            "repeat": 0,
            "latency_s": 1.0,
            "output_text": "a",
            "error": None,
        },
        {
            "item_id": "mt-000",
            "beam": 1,
            "repeat": 1,
            "latency_s": 1.2,
            "output_text": "a",
            "error": None,
        },
        {
            "item_id": "mt-000",
            "beam": 1,
            "repeat": 2,
            "latency_s": 9.0,
            "output_text": "a",
            "error": None,
        },
    ]

    records = per_item_medians(rows, items, 1)

    assert records[0]["result"]["latency_s"] == pytest.approx(1.2)
    assert records[0]["repeats"] == 3
    assert records[0]["beam_size"] == 1
    # Envelope stays compatible with scripts/summarise_beam_sweep.py
    assert records[0]["item"]["id"] == "mt-000"


def test_kv_cache_scales_with_beam_width():
    one = kv_cache_mb("opus-mt-vi-en-ct2-cpu", 1)
    four = kv_cache_mb("opus-mt-vi-en-ct2-cpu", 4)
    eight = kv_cache_mb("opus-mt-vi-en-ct2-cpu", 8)

    assert one is not None
    assert four == pytest.approx(one * 4)
    assert eight == pytest.approx(one * 8)
    # 2 (K,V) x 6 layers x 512 d_model x 256 tokens x 4 B = 6.0 MB per hypothesis
    assert one == pytest.approx(6.0, abs=0.01)


def test_kv_cache_is_none_for_unknown_geometry():
    assert kv_cache_mb("some-unpinned-candidate", 4) is None


def test_bleu_reports_both_definitions_and_a_signature():
    refs = ["the quick brown fox jumps over the lazy dog today"]
    scores = bleu_scores(refs, refs)

    assert scores["sentence_bleu"] == 100.0
    assert scores["corpus_bleu"] == 100.0
    assert "tok:13a" in scores["signature"]
    assert scores["chrf"] == 100.0


def test_monotonic_breaks_flags_a_wider_beam_finishing_faster():
    items = [make_item(0)]
    rows = [
        {
            "item_id": "mt-000",
            "beam": 1,
            "repeat": 0,
            "latency_s": 1.0,
            "output_text": "a",
            "error": None,
        },
        {
            "item_id": "mt-000",
            "beam": 4,
            "repeat": 0,
            "latency_s": 0.5,
            "output_text": "a",
            "error": None,
        },
    ]

    breaks = monotonic_breaks(rows, items, [1, 4])

    assert breaks["transitions"] == 1
    assert breaks["breaks"] == 1
    assert breaks["example"]


def test_summarise_keeps_quality_latency_and_kv_columns_separate():
    items = [make_item(0), make_item(1)]
    rows = []
    for beam, latency in ((1, 1.0), (4, 2.0)):
        for index in range(2):
            for repeat in range(2):
                rows.append(
                    {
                        "item_id": f"mt-{index:03d}",
                        "beam": beam,
                        "repeat": repeat,
                        "latency_s": latency,
                        "output_text": items[index].reference_text,
                        "error": None,
                    }
                )

    summary = summarise(rows, items, [1, 4], "opus-mt-vi-en-ct2-cpu")

    first, second = summary["beams"]
    assert first["latency_ratio"] == pytest.approx(1.0)
    assert second["latency_ratio"] == pytest.approx(2.0)
    assert first["kv_cache_mb"] == pytest.approx(6.0, abs=0.01)
    assert second["kv_cache_mb"] == pytest.approx(24.0, abs=0.01)
    assert first["corpus_bleu"] == 100.0
    assert first["signature"].startswith("nrefs:1")
    assert summary["items"] == 2


def test_summarise_reports_p50_and_p95_from_the_repeat_population():
    items = [make_item(i) for i in range(4)]
    rows = []
    for index, latency in enumerate((1.0, 2.0, 3.0, 4.0)):
        rows.append(
            {
                "item_id": f"mt-{index:03d}",
                "beam": 1,
                "repeat": 0,
                "latency_s": latency,
                "output_text": items[index].reference_text,
                "error": None,
            }
        )

    row = summarise(rows, items, [1], "fake")["beams"][0]

    assert row["latency_p50_s"] == pytest.approx(
        statistics.median([1.0, 2.0, 3.0, 4.0])
    )
    assert row["latency_p95_s"] >= row["latency_p50_s"]


def test_mt_items_rejects_a_manifest_without_mt_rows():
    manifest = RunManifest(
        items=[EvalItem(id="asr-1", stage="ASR", language="vi", audio_ref="a.wav")]
    )

    with pytest.raises(ValueError, match="no MT items"):
        mt_items(manifest, None)


def test_mt_items_filters_by_stage_and_honours_limit():
    manifest = RunManifest(
        items=[
            EvalItem(id="asr-1", stage="ASR", language="vi", audio_ref="a.wav"),
            make_item(0),
            make_item(1),
        ]
    )

    assert [item.id for item in mt_items(manifest, None)] == ["mt-000", "mt-001"]
    assert [item.id for item in mt_items(manifest, 1)] == ["mt-000"]
