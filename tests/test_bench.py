"""Model-free tests for the v0 bench harness (no Whisper/Opus/Piper weights).

Covers the eval-manifest schema round-trip, the off-device scorer on known
inputs, and the FLEURS id-merging logic (audio<->transcript must pair by id
even when the parquet's physical row order is shuffled).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bench.adapters import Candidate
from bench.registry import REGISTRY, default_candidate_id_for_stage
from bench.schema import EvalItem, RunManifest, StageResult
from bench.scorer import score_item


def test_manifest_roundtrip(tmp_path: Path) -> None:
    items = [
        EvalItem(
            id="a",
            stage="MT",
            language="vi",
            direction="vi->en",
            input_text="Xin chào",
            reference_text="Hello",
        ),
        EvalItem(
            id="b",
            stage="ASR",
            language="en",
            audio_ref="/tmp/x.wav",
            reference_text="hi",
            snr=5.0,
            noise_type="steady",
        ),
    ]
    m = RunManifest(version="eval_manifest_v1", items=items)
    out = tmp_path / "eval_manifest_v1.json"
    m.to_json(out)
    back = RunManifest.from_json(out)
    assert back.version == m.version
    assert [asdict(i) for i in back.items] == [asdict(i) for i in m.items]


def test_scorer_wer_cer_exact_and_off_by_one() -> None:
    item = EvalItem(id="a", stage="ASR", language="en", reference_text="the cat sat")
    exact = score_item(
        item, StageResult("c", "a", "ASR", output_text="the cat sat"), None
    )
    assert exact.wer == 0.0
    assert exact.cer == 0.0
    off = score_item(
        item, StageResult("c", "a", "ASR", output_text="the dog sat"), None
    )
    assert off.wer == pytest.approx(1 / 3)
    assert off.cer is not None and off.cer > 0.0


def test_scorer_bleu_positive() -> None:
    # Reference needs >=4 tokens so 4-gram precision is defined (else BLEU=0).
    item = EvalItem(
        id="m",
        stage="MT",
        language="vi",
        direction="vi->en",
        input_text="Xin chào",
        reference_text="the cat sat on the mat",
    )
    met = score_item(
        item, StageResult("c", "m", "MT", output_text="the cat sat on the mat"), None
    )
    assert met.bleu is not None and met.bleu > 0.0


def test_scorer_skips_without_reference() -> None:
    item = EvalItem(id="a", stage="ASR", language="en")  # no reference_text
    met = score_item(item, StageResult("c", "a", "ASR", output_text="anything"), None)
    assert met.wer is None
    assert any("skipped" in n for n in met.notes)


def test_scorer_error_short_circuits() -> None:
    item = EvalItem(id="a", stage="ASR", language="en", reference_text="x")
    met = score_item(item, StageResult("c", "a", "ASR", error="Boom"), None)
    assert met.wer is None
    assert met.notes and met.notes[0].startswith("error")


def _write_synthetic_fleurs(
    path: Path, n: int = 100, shuffle_seed: int = 7
) -> dict[str, int]:
    import random

    expected: dict[str, int] = {}
    rows = []
    for i in range(n):
        rid = f"utt_{i:04d}"
        rows.append(
            {
                "id": rid,
                "transcription": f"text_{i}",
                "audio": {
                    "array": [float(i), float(i)],
                    "bytes": None,
                    "sampling_rate": 16000,
                },
            }
        )
        expected[rid] = i
    random.Random(shuffle_seed).shuffle(rows)  # physical order != id order
    audio_struct = pa.array(
        [
            {"array": r["audio"]["array"], "bytes": None, "sampling_rate": 16000}
            for r in rows
        ],
        type=pa.struct(
            [
                ("array", pa.list_(pa.float64())),
                ("bytes", pa.binary()),
                ("sampling_rate", pa.int64()),
            ]
        ),
    )
    tbl = pa.table(
        {
            "id": pa.array([r["id"] for r in rows]),
            "transcription": pa.array([r["transcription"] for r in rows]),
            "audio": audio_struct,
        }
    )
    pq.write_table(tbl, str(path), row_group_size=50)  # 2 row groups
    return expected


def test_fleurs_load_pairs_audio_to_transcript_by_id(tmp_path: Path) -> None:
    from bench.data_prep import _load_fleurs

    pq_path = tmp_path / "synthetic_fleurs.parquet"
    expected = _write_synthetic_fleurs(pq_path)
    out = _load_fleurs(str(pq_path), n=30, sample_rate=16000, seed=42)
    assert len(out) == 30
    for uid, tx, audio in out:
        i = expected[uid]
        assert abs(audio[0, 0].item() - i) < 1e-6
        assert tx == f"text_{i}"


def _write_synthetic_mt_parquets(
    dir_path: Path, n_vi: int = 40, n_en: int = 35
) -> None:
    """Write synthetic vi/en FLEURS parquets with overlapping ids."""
    import pyarrow as pa

    vi_rows = [
        {"id": f"{1000 + i}", "transcription": f"van ban tieng viet {i}"}
        for i in range(n_vi)
    ]
    en_rows = [
        {"id": f"{1000 + i}", "transcription": f"vietnamese text {i}"}
        for i in range(n_en)
    ]
    for rows, name in (
        (vi_rows, "fleurs_vi_vn_test.parquet"),
        (en_rows, "fleurs_en_us_test.parquet"),
    ):
        tbl = pa.table(
            {
                "id": pa.array([r["id"] for r in rows]),
                "transcription": pa.array([r["transcription"] for r in rows]),
            }
        )
        pq.write_table(tbl, str(dir_path / name))


def test_build_mt_manifest_full_pool_and_subsample(tmp_path: Path) -> None:
    from bench.data_prep import build_mt_manifest
    from bench.schema import RunManifest

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_synthetic_mt_parquets(raw)  # 35 overlapping ids (n_en < n_vi)
    out = tmp_path / "mt.json"

    build_mt_manifest(str(out), workdir=str(tmp_path))
    m = RunManifest.from_json(str(out))
    assert len(m.items) == 35
    ids = [it.id for it in m.items]
    assert ids == sorted(ids) and len(set(ids)) == 35
    it = m.items[0]
    assert it.stage == "MT" and it.direction == "vi->en" and it.language == "vi"
    assert it.input_text.startswith("van ban") and it.reference_text.startswith(
        "vietnamese"
    )

    # deterministic subsample (even spacing starts at round(step/2) = index 2)
    build_mt_manifest(str(out), workdir=str(tmp_path), n_items=10)
    m2 = RunManifest.from_json(str(out))
    assert len(m2.items) == 10
    assert m2.items[0].id == "vi-en-mt-1002"


# --- Registration of new benchmark candidates (M2M, Moonshine, Zipformer) ---

NEW_CANDIDATE_IDS = {
    "m2m100-vi-en-ct2-cpu": "MT",
    "moonshine-tiny-vi-hf-cpu": "ASR",
    "moonshine-tiny-en-hf-cpu": "ASR",
    "zipformer-vi-30m-sherpa-onnx-cpu": "ASR",
    "zipformer-en-sherpa-onnx-cpu": "ASR",
}

# Hy-MT runs via HF Transformers (issue #91: CT2 cannot convert it). Registered
# like any other candidate; the model itself is never loaded in tests.
HYMT_CANDIDATE_ID = "hy-mt1.5-1.8b-hf-cpu"


def test_new_candidates_registered() -> None:
    """All 5 ported candidates are registered with the right stage."""
    for cid, stage in NEW_CANDIDATE_IDS.items():
        assert cid in REGISTRY, f"{cid} missing from REGISTRY"
        cls, _, cfg = REGISTRY[cid]
        assert cls.id == cid
        assert cls.stage == stage
        assert isinstance(cfg, dict)


def test_candidate_ids_unique() -> None:
    assert len(REGISTRY) == len(set(REGISTRY))


def test_hymt_candidate_registered() -> None:
    """Hy-MT is registered as an MT candidate (model-free; no weights load)."""
    assert HYMT_CANDIDATE_ID in REGISTRY
    cls, _, cfg = REGISTRY[HYMT_CANDIDATE_ID]
    assert cls.id == HYMT_CANDIDATE_ID
    assert cls.stage == "MT"
    assert cfg == {"beam_size": 5}


def test_hymt_candidate_prompt_format() -> None:
    """XX->XX prompt matches the model card template (no ZH path)."""
    from bench.candidates.hy_mt_hf import HyMT15MTCandidate

    # static check: no module-level model load happens at import time
    assert HyMT15MTCandidate.stage == "MT"
    assert HyMT15MTCandidate.id == HYMT_CANDIDATE_ID


def test_stage_defaults_unchanged() -> None:
    """D2: Whisper (ASR) and Opus-MT (MT) remain the stage defaults."""
    assert default_candidate_id_for_stage("ASR") == "whisper-small-faster-whisper-cpu"
    assert default_candidate_id_for_stage("MT") == "opus-mt-vi-en-ct2-cpu"
    assert default_candidate_id_for_stage("TTS") == "piper-en-lessac-cpu"


def test_zipformer_beam_decoding_mapping() -> None:
    from bench.candidates.zipformer_asr import decoding_method_for_beam

    assert decoding_method_for_beam(1) == "greedy_search"
    for beam in (2, 4, 5, 8):
        assert decoding_method_for_beam(beam) == "modified_beam_search"


# --- run_manifest regression: --candidate override, no-filter, --config-override ---


class _FakeCandidate(Candidate):
    """Weight-free stand-in so run_manifest tests never touch a real model."""

    def __init__(self, cid: str = "fake-cand") -> None:
        self.id = cid

    def _infer(self, item: EvalItem) -> tuple[str | None, str | None]:
        return "fake output", None


def _fake_manifest() -> RunManifest:
    return RunManifest(
        version="eval_manifest_v1",
        items=[
            EvalItem(id="a", stage="ASR", language="vi", reference_text="x"),
            EvalItem(
                id="b",
                stage="MT",
                language="vi",
                direction="vi->en",
                input_text="y",
                reference_text="z",
            ),
        ],
    )


def test_run_manifest_no_filter_processes_all(monkeypatch, tmp_path: Path) -> None:
    """Regression for the fba5a2b filter bug: no --candidate must run every item."""
    from bench import run as bench_run

    def fake_build(item):  # noqa: ANN001
        return _FakeCandidate()

    monkeypatch.setattr(bench_run, "build_candidate", fake_build)
    monkeypatch.setattr(
        bench_run, "default_candidate_id_for_stage", lambda stage: "fake-cand"
    )
    records = bench_run.run_manifest(_fake_manifest(), tmp_path, None)
    assert len(records) == 2
    assert all(r["result"]["candidate_id"] == "fake-cand" for r in records)
    assert all(r["result"]["error"] is None for r in records)


def test_run_manifest_candidate_override(monkeypatch, tmp_path: Path) -> None:
    """--candidate overrides item.candidate_id (6898497 semantics)."""
    from bench import run as bench_run

    seen: list[str | None] = []

    def fake_build(item):  # noqa: ANN001
        seen.append(item.candidate_id)
        return _FakeCandidate(cid="fake-2")

    monkeypatch.setattr(bench_run, "build_candidate", fake_build)
    records = bench_run.run_manifest(_fake_manifest(), tmp_path, "fake-2")
    assert len(records) == 2
    assert all(r["result"]["candidate_id"] == "fake-2" for r in records)
    # build_candidate runs once per cid (cached); the item it saw must have the
    # overridden candidate_id (without the override it would be None -> default).
    assert seen == ["fake-2"]


def test_run_manifest_config_override(monkeypatch, tmp_path: Path) -> None:
    """--config-override reaches build_candidate via item.config."""
    from bench import run as bench_run

    captured: dict = {}

    def fake_build(item):  # noqa: ANN001
        captured["config"] = dict(item.config or {})
        return _FakeCandidate()

    monkeypatch.setattr(bench_run, "build_candidate", fake_build)
    monkeypatch.setattr(
        bench_run, "default_candidate_id_for_stage", lambda stage: "fake-cand"
    )
    bench_run.run_manifest(_fake_manifest(), tmp_path, None, {"beam_size": 7})
    assert captured["config"]["beam_size"] == 7


def test_run_manifest_config_override_preserves_existing_keys(
    monkeypatch, tmp_path: Path
) -> None:
    """--config-override merges: unrelated existing keys survive the override.

    The test above only proves a new key reaches the candidate. The contract that
    actually matters is ``{**item.config, **override}``: keys the override does
    not mention must be preserved, and keys it mentions must win.
    """
    from bench import run as bench_run

    captured: dict = {}

    def fake_build(item):  # noqa: ANN001
        captured["config"] = dict(item.config or {})
        return _FakeCandidate()

    manifest = _fake_manifest()
    for item in manifest.items:
        item.config = {"beam_size": 2, "some_existing_option": "keep"}

    monkeypatch.setattr(bench_run, "build_candidate", fake_build)
    monkeypatch.setattr(
        bench_run, "default_candidate_id_for_stage", lambda stage: "fake-cand"
    )
    bench_run.run_manifest(manifest, tmp_path, None, {"beam_size": 7})
    assert captured["config"] == {"beam_size": 7, "some_existing_option": "keep"}


# --- winterSolstice review on #124: deterministic model pick, distinct sampling ---


def test_resolve_model_file_prefers_int8_over_fp32(tmp_path: Path) -> None:
    """Two matching builds must resolve to the int8 one, not to glob order.

    The trained -epoch-N-avg-N exports ship an int8/fp32 pair for the same
    component. Path.glob order is filesystem-dependent, so picking matches[0]
    would let the filesystem decide which model is benchmarked.
    """
    from bench.candidates.zipformer_asr import _resolve_model_file

    (tmp_path / "decoder-epoch-99-avg-1.onnx").write_bytes(b"fp32")
    (tmp_path / "decoder-epoch-99-avg-1.int8.onnx").write_bytes(b"int8")

    picked = _resolve_model_file(tmp_path, "decoder")
    assert picked.name == "decoder-epoch-99-avg-1.int8.onnx"


def test_resolve_model_file_single_match(tmp_path: Path) -> None:
    from bench.candidates.zipformer_asr import _resolve_model_file

    (tmp_path / "encoder.int8.onnx").write_bytes(b"x")
    assert _resolve_model_file(tmp_path, "encoder").name == "encoder.int8.onnx"


def test_resolve_model_file_raises_when_genuinely_ambiguous(tmp_path: Path) -> None:
    """Two different epochs is not a precision choice — refuse rather than guess."""
    from bench.candidates.zipformer_asr import _resolve_model_file

    (tmp_path / "encoder-epoch-1-avg-1.int8.onnx").write_bytes(b"a")
    (tmp_path / "encoder-epoch-2-avg-1.int8.onnx").write_bytes(b"b")

    with pytest.raises(ValueError, match="several encoder"):
        _resolve_model_file(tmp_path, "encoder")


def test_resolve_model_file_missing_raises(tmp_path: Path) -> None:
    from bench.candidates.zipformer_asr import _resolve_model_file

    with pytest.raises(FileNotFoundError):
        _resolve_model_file(tmp_path, "encoder")


def test_evenly_spaced_indices_are_distinct_when_k_approaches_n() -> None:
    """Regression: rounding + clamping used to collapse picks into duplicates."""
    from bench.data_prep import _evenly_spaced_indices

    for n_available in (5, 7, 12, 40):
        for k in range(1, n_available + 1):
            picks = _evenly_spaced_indices(n_available, k)
            assert len(picks) == k, f"n={n_available} k={k}"
            assert len(set(picks)) == k, f"n={n_available} k={k} produced duplicates"
            assert all(0 <= p < n_available for p in picks)
            assert picks == sorted(picks)


def test_evenly_spaced_indices_rejects_k_larger_than_available() -> None:
    from bench.data_prep import _evenly_spaced_indices

    with pytest.raises(ValueError):
        _evenly_spaced_indices(5, 6)


def test_evenly_spaced_indices_rejects_non_positive_k() -> None:
    from bench.data_prep import _evenly_spaced_indices

    with pytest.raises(ValueError):
        _evenly_spaced_indices(5, 0)
