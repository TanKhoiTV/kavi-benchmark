"""Minimal test for one noisy case (real MUSAN + RIRS + 10 dB)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bench.vad_noise import (
    LEAD_SAMPLES,
    TRAIL_SAMPLES,
    align_noise,
    build_clip_audio,
    build_timeline_audio,
    frame_peak_p99,
    normalize_level,
    rms,
    run_case,
    speech_mask,
)


def test_align_noise_longer():
    arr = np.arange(100)
    out = align_noise(arr, 30)
    assert len(out) == 30
    assert np.array_equal(out, arr[:30])


def test_align_noise_shorter():
    arr = np.arange(10)
    out = align_noise(arr, 25)
    assert len(out) == 25
    assert np.array_equal(out[:10], arr)


def test_align_noise_equal():
    arr = np.arange(20)
    out = align_noise(arr, 20)
    assert len(out) == 20
    assert np.array_equal(out, arr)


REAL_NOISY_CASE_FILES = (
    "eval_data/vad_audio/fleurs/vi/vi_1660.wav",
    "assets/noise/musan/steady/noise-free-sound-0030.wav",
    "assets/noise/rirs/air_type1_air_binaural_aula_carolina_1_1_90_3.wav",
)


@pytest.mark.skipif(
    not all(Path(p).exists() for p in REAL_NOISY_CASE_FILES),
    reason="needs the materialised FLEURS clip, MUSAN noise and RIRS file",
)
def test_noisy_output() -> None:
    out = "/tmp/kavi_vad_test_noisy.wav"
    if Path(out).exists():
        Path(out).unlink()
    res = run_case(out_path=out)
    assert Path(out).exists(), "no output WAV"
    info = sf.info(out)
    assert info.samplerate == 16000, f"sr={info.samplerate}"
    assert info.frames > 0, "empty audio"
    assert res["target_snr_db"] == 10.0
    # Tolerance ±0.5 dB
    assert abs(res["measured_after_mix_dB"] - 10.0) <= 0.5, (
        f"SNR error={abs(res['measured_after_mix_dB'] - 10.0):.2f} dB"
    )
    assert not res["clipping"], "clipping occurred"
    print(
        f"target={res['target_snr_db']} measured_after_mix={res['measured_after_mix_dB']:.2f} peak={res['mixed_peak']:.4f} clipping={res['clipping']}"
    )


def _speech(n: int, seed: int) -> np.ndarray:
    return (0.1 * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


def test_timeline_clean_gt_and_silence() -> None:
    a, b = _speech(3200, 1), _speech(4800, 2)
    built = build_timeline_audio(a, b, 350)
    pause = 350 * 16
    assert built.gt_segments == [
        (LEAD_SAMPLES, LEAD_SAMPLES + 3200),
        (LEAD_SAMPLES + 3200 + pause, LEAD_SAMPLES + 3200 + pause + 4800),
    ]
    assert len(built.audio) == LEAD_SAMPLES + 3200 + pause + 4800 + TRAIL_SAMPLES
    assert np.all(built.audio[:LEAD_SAMPLES] == 0)
    assert np.all(built.audio[-TRAIL_SAMPLES:] == 0)
    assert built.noise_component is None
    assert built.snr_db_construction is None


@pytest.mark.parametrize("snr_db", [15.0, 5.0, 0.0, -5.0])
def test_timeline_noise_covers_whole_timeline_and_hits_snr(snr_db: float) -> None:
    a, b = _speech(8000, 3), _speech(8000, 4)
    noise = np.random.default_rng(5).standard_normal(5000).astype(np.float32)
    built = build_timeline_audio(a, b, 200, noise=noise, snr_db=snr_db)
    audio = built.audio.astype(np.float64)
    assert rms(audio[:LEAD_SAMPLES]) > 0
    assert rms(audio[-TRAIL_SAMPLES:]) > 0
    (_, a_end), (b_start, _) = built.gt_segments
    assert rms(audio[a_end:b_start]) > 0
    residual = audio - built.reference
    mask = speech_mask(len(audio), built.gt_segments)
    measured = 20 * np.log10(rms(built.reference[mask]) / rms(residual))
    assert abs(measured - snr_db) < 0.01


def test_timeline_rir_tail_spills_into_pause() -> None:
    a, b = _speech(3200, 6), _speech(3200, 7)
    rir = np.exp(-np.arange(1600) / 200.0).astype(np.float32)
    built = build_timeline_audio(a, b, 200, rir=rir)
    (_, a_end), (b_start, _) = built.gt_segments
    assert np.max(np.abs(built.reference[a_end : a_end + 800])) > 0
    assert b_start - a_end == 200 * 16


def test_timeline_identity_rir_keeps_speech() -> None:
    a, b = _speech(3200, 8), _speech(3200, 9)
    plain = build_timeline_audio(a, b, 200)
    ident = build_timeline_audio(a, b, 200, rir=np.array([1.0], dtype=np.float32))
    assert np.allclose(plain.audio, ident.audio, atol=1e-6)


def test_timeline_peak_limited_keeps_snr() -> None:
    a, b = 4 * _speech(3200, 10), 4 * _speech(3200, 11)
    noise = np.random.default_rng(12).standard_normal(4000).astype(np.float32)
    built = build_timeline_audio(
        a, b, 200, noise=noise, snr_db=-5.0, target_p99_peak=None
    )
    assert built.peak_limited
    assert float(np.max(np.abs(built.audio))) <= 0.99 + 1e-6
    residual = built.audio.astype(np.float64) - built.reference
    mask = speech_mask(len(built.audio), built.gt_segments)
    measured = 20 * np.log10(rms(built.reference[mask]) / rms(residual))
    assert abs(measured + 5.0) < 0.01


def test_timeline_noise_requires_snr() -> None:
    a, b = _speech(1600, 13), _speech(1600, 14)
    with pytest.raises(ValueError):
        build_timeline_audio(a, b, 200, noise=np.ones(100, dtype=np.float32))


def test_normalize_level_hits_target() -> None:
    quiet = 0.002 * np.random.default_rng(20).standard_normal(32000)
    out, gain = normalize_level(quiet.astype(np.float32), 0.35)
    assert frame_peak_p99(out) == pytest.approx(0.35, rel=1e-4)
    assert gain == pytest.approx(0.35 / frame_peak_p99(quiet), rel=1e-4)


def test_normalize_level_rejects_near_silence() -> None:
    with pytest.raises(ValueError):
        normalize_level(np.full(3200, 1e-6, dtype=np.float32))


def test_timeline_balances_weak_and_strong_clips() -> None:
    weak = 0.003 * _speech(32000, 21) / 0.1
    strong = 0.8 * _speech(32000, 22) / 0.1
    built = build_timeline_audio(weak, strong, 200)
    (a0, a1), (b0, b1) = built.gt_segments
    level_a = frame_peak_p99(built.audio[a0:a1])
    level_b = frame_peak_p99(built.audio[b0:b1])
    assert level_a == pytest.approx(0.35, rel=0.05)
    assert level_b == pytest.approx(0.35, rel=0.05)
    assert built.gain_a > 10 * built.gain_b


def test_timeline_target_none_keeps_source_level() -> None:
    a, b = _speech(3200, 23), _speech(3200, 24)
    built = build_timeline_audio(a, b, 200, target_p99_peak=None)
    (a0, a1), _ = built.gt_segments
    assert np.allclose(built.audio[a0:a1], a)
    assert (built.gain_a, built.gain_b) == (1.0, 1.0)


def test_timeline_snr_holds_after_level_normalisation() -> None:
    a, b = 0.004 * _speech(16000, 25) / 0.1, 0.9 * _speech(16000, 26) / 0.1
    noise = np.random.default_rng(27).standard_normal(8000).astype(np.float32)
    built = build_timeline_audio(a, b, 350, noise=noise, snr_db=5.0)
    residual = built.audio.astype(np.float64) - built.reference
    mask = speech_mask(len(built.audio), built.gt_segments)
    measured = 20 * np.log10(rms(built.reference[mask]) / rms(residual))
    assert abs(measured - 5.0) < 0.01


def test_timeline_bounds_trim_clips_and_set_gt() -> None:
    a, b = _speech(20000, 30), _speech(24000, 31)
    built = build_timeline_audio(
        a, b, 350, bounds_a=(2000, 12000), bounds_b=(1000, 9000), target_p99_peak=None
    )
    pause = 350 * 16
    assert built.gt_segments == [
        (LEAD_SAMPLES, LEAD_SAMPLES + 10000),
        (LEAD_SAMPLES + 10000 + pause, LEAD_SAMPLES + 10000 + pause + 8000),
    ]
    (a0, a1), (b0, b1) = built.gt_segments
    assert np.allclose(built.audio[a0:a1], a[2000:12000])
    assert np.allclose(built.audio[b0:b1], b[1000:9000])
    assert len(built.audio) == b1 + TRAIL_SAMPLES


def test_timeline_bounds_outside_clip_rejected() -> None:
    a, b = _speech(1000, 32), _speech(1000, 33)
    with pytest.raises(ValueError):
        build_timeline_audio(a, b, 200, bounds_a=(0, 5000))


def test_clip_audio_has_margins_and_hits_snr() -> None:
    speech = _speech(16000, 34)
    noise = np.random.default_rng(35).standard_normal(5000).astype(np.float32)
    built = build_clip_audio(speech, noise=noise, snr_db=5.0)
    assert built.gt_segments == [(8000, 24000)]
    assert len(built.audio) == 8000 + 16000 + 8000
    assert rms(built.audio[:8000].astype(np.float64)) > 0
    residual = built.audio.astype(np.float64) - built.reference
    mask = speech_mask(len(built.audio), built.gt_segments)
    measured = 20 * np.log10(rms(built.reference[mask]) / rms(residual))
    assert abs(measured - 5.0) < 0.01
    assert frame_peak_p99(built.reference[mask]) == pytest.approx(0.35, rel=0.05)


def test_clip_audio_clean_keeps_margins_silent() -> None:
    built = build_clip_audio(_speech(8000, 36))
    assert np.all(built.audio[:8000] == 0) and np.all(built.audio[-8000:] == 0)
