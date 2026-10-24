"""Tests for scripts.audit_vad_manifest using a small synthetic environment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bench.vad_manifest import build_vad_manifest
from scripts.audit_vad_manifest import audit_gold, audit_manifest


def _wav(path: Path, n: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = 0.1 * np.random.default_rng(seed).standard_normal(n)
    sf.write(str(path), x.astype("float32"), 16000)


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    files = {}
    for i in range(22):
        for kind in ("steady", "impulsive"):
            _wav(tmp_path / f"assets/noise/musan/{kind}/{kind[0]}{i}.wav", 4000, i)
        _wav(tmp_path / f"assets/noise/rirs/r{i}.wav", 1000, i)
        for lang in ("vi", "en"):
            key = f"eval_data/vad_audio/fleurs/{lang}/{lang}_{i}.wav"
            _wav(tmp_path / key, 20000, i)
            files[key] = {
                "onset": 1000 + i,
                "offset": 15000 + 7 * i,
                "n_samples": 20000,
            }
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench/vad_source_bounds.json").write_text(
        json.dumps({"files": files}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    build_vad_manifest(out_path=str(tmp_path / "m.json"))
    return tmp_path / "m.json"


def test_audit_accepts_a_freshly_built_manifest(built: Path) -> None:
    result = audit_manifest(str(built))
    assert result["problems"] == []
    assert result["total"] == 800
    assert result["unique_ids"] == 800
    assert result["by_language"] == {"vi": 400, "en": 400}
    assert result["cells"] == 40
    assert result["cells_off_size"] == 0 and result["cells_missing"] == 0


def test_audit_detects_ground_truth_that_ignores_bounds(built: Path) -> None:
    data = json.loads(built.read_text(encoding="utf-8"))
    data["items"][0]["ground_truth_segments"][0][1] += 100
    built.write_text(json.dumps(data), encoding="utf-8")
    problems = audit_manifest(str(built))["problems"]
    assert any("GT does not match" in p for p in problems)


def test_audit_detects_missing_bounds(built: Path) -> None:
    data = json.loads(built.read_text(encoding="utf-8"))
    data["items"][3]["speech_a_bounds"] = None
    built.write_text(json.dumps(data), encoding="utf-8")
    assert any(
        "missing source bounds" in p for p in audit_manifest(str(built))["problems"]
    )


def test_audit_detects_wrong_noise_mapping(built: Path) -> None:
    data = json.loads(built.read_text(encoding="utf-8"))
    street = next(i for i in data["items"] if i["condition"] == "street")
    street["rir_path"] = "assets/noise/rirs/r0.wav"
    built.write_text(json.dumps(data), encoding="utf-8")
    assert any(
        "RIR presence wrong" in p for p in audit_manifest(str(built))["problems"]
    )


def test_audit_gold_flags_wrong_size(tmp_path: Path) -> None:
    f = tmp_path / "gold.csv"
    f.write_text(
        "id,audio_path,onset_sample,offset_sample,condition,snr_db,annotator\n"
        "g1,/vi/a.wav,0,1,quiet,clean,x\n",
        encoding="utf-8",
    )
    problems = audit_gold(str(f))["problems"]
    assert any("expected 50" in p for p in problems)


def test_audit_checks_forty_items_per_cell(built: Path, tmp_path: Path) -> None:
    big = tmp_path / "m40.json"
    build_vad_manifest(out_path=str(big), n_per_lang=20)
    ok = audit_manifest(str(big), per_cell_per_lang=20)
    assert ok["problems"] == []
    assert ok["total"] == ok["expected_total"] == 1600
    assert ok["cells_off_size"] == 0
    wrong = audit_manifest(str(big), per_cell_per_lang=10)
    assert wrong["cells_off_size"] == 80
    assert wrong["total"] != wrong["expected_total"]
