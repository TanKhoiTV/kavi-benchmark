"""Tests for VAD manifest schema and builder; synthetic only, no real audio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bench.vad_manifest import (
    VadItem,
    VadManifest,
    build_gold_template,
    build_vad_manifest,
)


def test_manifest_schema_fields() -> None:
    item = VadItem(
        id="vad-test-01",
        language="vi",
        condition="quiet",
        clean_audio_path="eval_data/mixed/test.wav",
        noise_type="clean",
        pause_ms=500,
        ground_truth_segments=[(500, 2500)],
        energy_threshold=0.05,
        speech_timeout_ms=500,
    )
    manifest = VadManifest(items=[item])
    assert manifest.version == "vad_manifest_v2"
    assert len(manifest.items) == 1
    assert manifest.items[0].noise_type == "clean"
    assert manifest.items[0].pause_ms == 500
    assert manifest.items[0].condition == "quiet"


def test_manifest_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "vad_manifest_v2.json"
    manifest = VadManifest()
    manifest.items.append(
        VadItem(
            id="t1",
            language="vi",
            condition="quiet",
            clean_audio_path="a.wav",
            noise_type="clean",
            pause_ms=500,
            ground_truth_segments=[(500, 2500)],
        )
    )
    manifest.to_json(out)
    back = VadManifest.from_json(out)
    assert back.version == "vad_manifest_v2"
    assert [i.id for i in back.items] == ["t1"]
    assert back.items[0].condition == "quiet"
    assert back.items[0].pause_ms == 500


def test_seed_determinism() -> None:
    m1 = VadManifest(
        items=[
            VadItem(
                id="seed-a",
                language="en",
                condition="quiet",
                clean_audio_path="a.wav",
                noise_type="clean",
                pause_ms=500,
                ground_truth_segments=[(500, 2500)],
            )
        ]
    )
    m2 = VadManifest(
        items=[
            VadItem(
                id="seed-a",
                language="en",
                condition="quiet",
                clean_audio_path="a.wav",
                noise_type="clean",
                pause_ms=500,
                ground_truth_segments=[(500, 2500)],
            )
        ]
    )
    assert m1.items[0].id == m2.items[0].id


def test_build_stops_when_fleurs_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        build_vad_manifest(
            out_path=str(tmp_path / "vad_stop.json"),
            workdir=str(tmp_path),
            use_fleurs=True,
        )


@pytest.mark.skipif(
    not any(Path("eval_data/vad_audio/fleurs/vi").glob("*.wav"))
    or not any(Path("eval_data/vad_audio/fleurs/en").glob("*.wav")),
    reason="needs the materialised FLEURS WAVs",
)
def test_gold_template_exists_and_fields() -> None:
    build_gold_template(path="/tmp/vad_gold.csv")
    content = Path("/tmp/vad_gold.csv").read_text(encoding="utf-8")
    assert (
        "id,audio_path,onset_sample,offset_sample,condition,snr_db,annotator" in content
    )
    assert "gold-001" in content
    lines = content.strip().splitlines()
    assert len(lines) == 51  # header + 50 gold rows (stage 3A v2)
    # Verify 25 VI / 25 EN / coverage >=2 per cell handled in build_gold_template


def test_snr_tolerance_declaration() -> None:
    item = VadItem(
        id="test",
        language="vi",
        condition="street",
        clean_audio_path="x.wav",
        noise_type="steady",
        snr_db=10.0,
        pause_ms=500,
    )
    assert item.snr_db == 10.0
    assert item.condition == "street"
    # Stage-0 reference: data_prep._mix uses torchaudio.functional.add_noise with target SNR


def test_vad_item_accepts_valid_condition_snr():
    item = VadItem(
        id="vi-street-10-200-000",
        language="vi",
        condition="street",
        clean_audio_path="clean.wav",
        noise_type="impulsive",
        noise_path="noise.wav",
        snr_db=10.0,
        pause_ms=200,
    )
    assert item.condition == "street"
    assert item.snr_db == 10.0
    assert item.pause_ms == 200


def test_vad_item_rejects_invalid_condition_snr():
    with pytest.raises(ValueError):
        VadItem(
            id="vi-street-0-200-000",
            language="vi",
            condition="street",
            clean_audio_path="clean.wav",
            noise_type="impulsive",
            noise_path="noise.wav",
            snr_db=0.0,
            pause_ms=200,
        )


def test_vad_item_rejects_invalid_pause():
    with pytest.raises(ValueError):
        VadItem(
            id="vi-street-10-100-000",
            language="vi",
            condition="street",
            clean_audio_path="clean.wav",
            noise_type="impulsive",
            noise_path="noise.wav",
            snr_db=10.0,
            pause_ms=100,
        )


def test_vad_item_accepts_far_field_minus_5_db():
    item = VadItem(
        id="en-far-field-m5-700-000",
        language="en",
        condition="far-field",
        clean_audio_path="clean.wav",
        noise_type="impulsive",
        noise_path="noise.wav",
        snr_db=-5.0,
        pause_ms=700,
    )
    assert item.condition == "far-field"
    assert item.snr_db == -5.0
    assert item.pause_ms == 700


def test_vad_item_accepts_quiet_clean():
    item = VadItem(
        id="vi-quiet-clean-200-000",
        language="vi",
        condition="quiet",
        clean_audio_path="clean.wav",
        noise_type="clean",
        noise_path=None,
        snr_db=None,
        rir_path=None,
        pause_ms=200,
    )
    assert item.condition == "quiet"
    assert item.noise_type == "clean"
    assert item.snr_db is None


BOUNDS_FILE = "bench/vad_source_bounds.json"


def _write_wav(path: Path, n: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = 0.1 * np.random.default_rng(seed).standard_normal(n)
    sf.write(str(path), x.astype("float32"), 16000)


def _fake_env(root: Path, with_bounds: bool = True) -> dict[str, tuple[int, int]]:
    for i in range(12):
        for kind in ("steady", "impulsive"):
            _write_wav(root / f"assets/noise/musan/{kind}/{kind[0]}{i}.wav", 4000, i)
        _write_wav(root / f"assets/noise/rirs/r{i}.wav", 1000, i)
        for lang in ("vi", "en"):
            _write_wav(
                root / f"eval_data/vad_audio/fleurs/{lang}/{lang}_{i}.wav", 20000, i
            )
    bounds = {}
    files = {}
    for lang in ("vi", "en"):
        for i in range(12):
            key = f"eval_data/vad_audio/fleurs/{lang}/{lang}_{i}.wav"
            bounds[key] = (1000 + i, 15000 + 7 * i)
            files[key] = {
                "onset": 1000 + i,
                "offset": 15000 + 7 * i,
                "n_samples": 20000,
            }
    if with_bounds:
        (root / "bench").mkdir(exist_ok=True)
        (root / BOUNDS_FILE).write_text(json.dumps({"files": files}), encoding="utf-8")
    return bounds


def test_builder_ground_truth_uses_reference_bounds(tmp_path, monkeypatch) -> None:
    bounds = _fake_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    manifest = build_vad_manifest(out_path=str(tmp_path / "m.json"))
    assert len(manifest.items) == 800
    for item in manifest.items:
        a = bounds[item.clean_audio_path]
        b = bounds[item.speech_b_path]
        assert tuple(item.speech_a_bounds) == a
        assert tuple(item.speech_b_bounds) == b
        a_len, b_len = a[1] - a[0], b[1] - b[0]
        pause = item.pause_ms * 16
        assert item.ground_truth_segments == [
            (16000, 16000 + a_len),
            (16000 + a_len + pause, 16000 + a_len + pause + b_len),
        ]


def test_builder_same_seed_gives_identical_files(tmp_path, monkeypatch) -> None:
    _fake_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    build_vad_manifest(out_path=str(tmp_path / "a.json"), seed=42)
    build_vad_manifest(out_path=str(tmp_path / "b.json"), seed=42)
    build_vad_manifest(out_path=str(tmp_path / "c.json"), seed=43)
    a = (tmp_path / "a.json").read_bytes()
    assert a == (tmp_path / "b.json").read_bytes()
    assert a != (tmp_path / "c.json").read_bytes()


def test_builder_roundtrip_keeps_bounds(tmp_path, monkeypatch) -> None:
    _fake_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    build_vad_manifest(out_path=str(tmp_path / "m.json"))
    back = VadManifest.from_json(tmp_path / "m.json")
    assert all(i.speech_a_bounds is not None for i in back.items)


def test_builder_requires_bounds_file(tmp_path, monkeypatch) -> None:
    _fake_env(tmp_path, with_bounds=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="vad-bounds"):
        build_vad_manifest(out_path=str(tmp_path / "m.json"))


def test_builder_stops_when_a_selected_clip_has_no_bounds(
    tmp_path, monkeypatch
) -> None:
    _fake_env(tmp_path)
    path = tmp_path / BOUNDS_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data["files"].pop("eval_data/vad_audio/fleurs/vi/vi_3.wav")
    data["files"].pop("eval_data/vad_audio/fleurs/vi/vi_4.wav")
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for seed in range(40):
        try:
            build_vad_manifest(out_path=str(tmp_path / "m.json"), seed=seed)
        except FileNotFoundError as exc:
            assert "No reference speech bounds" in str(exc)
            return
    raise AssertionError("no seed selected a clip without bounds")
