"""Minimal noisy-audio pipeline for one Stage 3A case (1 clean + MUSAN + RIRS + 10 dB)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy import signal

SR = 16000
LEAD_SAMPLES = 16000
TRAIL_SAMPLES = 16000
PEAK_LIMIT = 0.99
LEVEL_FRAME_SAMPLES = 160
TARGET_P99_PEAK = 0.35
MIN_P99_PEAK = 1e-4


def load_mono(path: str, sr: int = SR) -> np.ndarray:
    data, fs = sf.read(path, dtype="float32", always_2d=True)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if fs != sr:
        import torch
        from torchaudio import functional

        t = torch.from_numpy(data).unsqueeze(0)
        t = functional.resample(t, fs, sr)
        data = t.squeeze(0).numpy()
    return data


def apply_rir(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    out = signal.fftconvolve(speech, rir, mode="full")
    out = out[: len(speech)]
    peak = np.max(np.abs(out))
    if peak > 0.9:
        out = out * (0.9 / peak)
    return out


def align_noise(noise: np.ndarray, target_len: int) -> np.ndarray:
    if len(noise) >= target_len:
        return noise[:target_len]
    repeats = (target_len + len(noise) - 1) // len(noise)
    aligned = np.tile(noise, repeats)
    return aligned[:target_len]


def mix_snr(
    speech: np.ndarray, noise: np.ndarray, snr_db: float
) -> tuple[np.ndarray, float, float, bool]:
    # 1. Align noise before RMS
    aligned_noise = align_noise(noise, len(speech))
    # 2. Compute RMS on aligned components
    s_rms = float(np.sqrt(np.mean(speech**2)))
    n_rms_before = float(np.sqrt(np.mean(aligned_noise**2)))
    if s_rms == 0:
        raise ValueError("Speech RMS is zero")
    # 3. Scale noise to target SNR
    target_noise_rms = s_rms / (10 ** (snr_db / 20.0))
    scale_factor = target_noise_rms / (n_rms_before + 1e-12)
    scaled_noise = aligned_noise * scale_factor
    # 4. Measure SNR before mix (signal / scaled_noise aligned)
    n_rms_after = float(np.sqrt(np.mean(scaled_noise**2)))
    measured_before = float(20 * np.log10(s_rms / (n_rms_after + 1e-12)))
    # 5. Mix
    mixed = speech + scaled_noise
    # 6. Check/report clipping
    peak = float(np.max(np.abs(mixed)))
    clipping = peak > 0.99
    if clipping:
        # Preserve scale but note clipping; do not silently rescale output and claim same SNR
        pass
    # Measure after mix from aligned components
    return mixed, float(measured_before), float(peak), float(clipping)


def measure_snr_from_components(speech: np.ndarray, noise: np.ndarray) -> float:
    s_rms = float(np.sqrt(np.mean(speech**2)))
    n_rms = float(np.sqrt(np.mean(noise**2)))
    return float(20 * np.log10(s_rms / (n_rms + 1e-12)))


def run_case(
    clean_path: str = "eval_data/vad_audio/fleurs/vi/vi_1660.wav",
    noise_path: str = "assets/noise/musan/steady/noise-free-sound-0030.wav",
    rir_path: str = "assets/noise/rirs/air_type1_air_binaural_aula_carolina_1_1_90_3.wav",
    snr_db: float = 10.0,
    out_path: str = "/tmp/kavi_vad_test_noisy.wav",
) -> dict:
    clean = load_mono(clean_path, SR)
    noise = load_mono(noise_path, SR)
    rir = load_mono(rir_path, SR)
    reverbed = apply_rir(clean, rir)
    noisy, measured_before, mixed_peak, clipping = mix_snr(reverbed, noise, snr_db)
    # After mix measurement from actual components (reverbed + scaled_noise)
    # Rebuild scaled_noise for measurement consistency
    aligned_noise = align_noise(noise, len(reverbed))
    s_rms = float(np.sqrt(np.mean(reverbed**2)))
    target_noise_rms = s_rms / (10 ** (snr_db / 20.0))
    scale_factor = target_noise_rms / (
        float(np.sqrt(np.mean(aligned_noise**2))) + 1e-12
    )
    scaled_noise = aligned_noise * scale_factor
    measured_after = measure_snr_from_components(reverbed, scaled_noise)
    sf.write(out_path, noisy, SR)
    return {
        "target_snr_db": snr_db,
        "measured_before_mix_dB": measured_before,
        "measured_after_mix_dB": measured_after,
        "mixed_peak": mixed_peak,
        "clipping": clipping,
        "output_path": out_path,
        "clean_path": clean_path,
        "noise_path": noise_path,
        "rir_path": rir_path,
        "sample_rate": SR,
        "duration_s": float(len(noisy) / SR),
        "speech_peak": float(np.max(np.abs(reverbed))),
        "noise_peak_before_scale": float(np.max(np.abs(aligned_noise))),
    }


@dataclass
class TimelineAudio:
    audio: np.ndarray
    reference: np.ndarray
    noise_component: np.ndarray | None
    gt_segments: list[tuple[int, int]]
    snr_db_construction: float | None
    peak_limited: bool
    gain_a: float = 1.0
    gain_b: float = 1.0


def frame_peak_p99(x: np.ndarray) -> float:
    a = np.abs(np.asarray(x, dtype=np.float64))
    n = len(a) // LEVEL_FRAME_SAMPLES
    if n == 0:
        return float(a.max()) if len(a) else 0.0
    peaks = a[: n * LEVEL_FRAME_SAMPLES].reshape(n, LEVEL_FRAME_SAMPLES).max(axis=1)
    return float(np.percentile(peaks, 99))


def normalize_level(
    x: np.ndarray, target_p99_peak: float = TARGET_P99_PEAK
) -> tuple[np.ndarray, float]:
    p99 = frame_peak_p99(x)
    if p99 < MIN_P99_PEAK:
        raise ValueError(f"Clip too quiet to normalise (p99 frame peak {p99:.2e})")
    gain = target_p99_peak / p99
    return (np.asarray(x, dtype=np.float64) * gain).astype(np.float32), float(gain)


def speech_mask(length: int, segments: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in segments:
        mask[start:end] = True
    return mask


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _cut(x: np.ndarray, bounds: tuple[int, int] | None) -> np.ndarray:
    if bounds is None:
        return x
    onset, offset = int(bounds[0]), int(bounds[1])
    if not 0 <= onset < offset <= len(x):
        raise ValueError(f"bounds {bounds} outside clip of {len(x)} samples")
    return x[onset:offset]


def build_timeline_audio(
    speech_a: np.ndarray,
    speech_b: np.ndarray,
    pause_ms: int,
    noise: np.ndarray | None = None,
    rir: np.ndarray | None = None,
    snr_db: float | None = None,
    target_p99_peak: float | None = TARGET_P99_PEAK,
    bounds_a: tuple[int, int] | None = None,
    bounds_b: tuple[int, int] | None = None,
) -> TimelineAudio:
    """Build lead + A + pause + B + trail, then apply RIR and noise to all of it.

    Bounds: when bounds_a / bounds_b are given as (onset, offset) sample indices
    into the source clip, the clip is cut to that span first, so the timeline holds
    only speech (reference-VAD boundaries) and pauses are real silences.

    Level: unless target_p99_peak is None, speech A and speech B are each scaled
    so the 99th percentile of their 10 ms frame peaks equals target_p99_peak,
    before the timeline is built. Absolute level is otherwise the FLEURS source
    level, which varies by about 100x between recordings.

    SNR definition: RMS of the reverberated clean timeline over the GT speech
    segments versus RMS of the aligned noise over the whole timeline.
    """
    if (noise is None) != (snr_db is None):
        raise ValueError("noise and snr_db must be given together")

    speech_a = _cut(speech_a, bounds_a)
    speech_b = _cut(speech_b, bounds_b)

    gain_a = gain_b = 1.0
    if target_p99_peak is not None:
        speech_a, gain_a = normalize_level(speech_a, target_p99_peak)
        speech_b, gain_b = normalize_level(speech_b, target_p99_peak)

    pause_samples = int(pause_ms * 16)
    a_start = LEAD_SAMPLES
    a_end = a_start + len(speech_a)
    b_start = a_end + pause_samples
    b_end = b_start + len(speech_b)
    total = b_end + TRAIL_SAMPLES

    timeline = np.zeros(total, dtype=np.float32)
    timeline[a_start:a_end] = speech_a
    timeline[b_start:b_end] = speech_b
    segments = [(a_start, a_end), (b_start, b_end)]
    return _render(timeline, segments, noise, rir, snr_db, gain_a, gain_b)


def _render(
    timeline: np.ndarray,
    segments: list[tuple[int, int]],
    noise: np.ndarray | None,
    rir: np.ndarray | None,
    snr_db: float | None,
    gain_a: float,
    gain_b: float,
) -> TimelineAudio:
    total = len(timeline)
    reference = apply_rir(timeline, rir) if rir is not None else timeline
    reference = np.asarray(reference, dtype=np.float64)

    noise_component = None
    snr_construction = None
    mixed = reference
    if noise is not None and snr_db is not None:
        aligned = np.asarray(align_noise(noise, total), dtype=np.float64)
        s_rms = rms(reference[speech_mask(total, segments)])
        n_rms = rms(aligned)
        if s_rms == 0 or n_rms == 0:
            raise ValueError("Zero RMS for SNR calculation")
        noise_component = aligned * (s_rms / (10 ** (snr_db / 20.0)) / n_rms)
        mixed = reference + noise_component
        snr_construction = float(20 * np.log10(s_rms / rms(noise_component)))

    peak = float(np.max(np.abs(mixed)))
    peak_limited = peak > PEAK_LIMIT
    if peak_limited:
        gain = PEAK_LIMIT / peak
        mixed = mixed * gain
        reference = reference * gain
        if noise_component is not None:
            noise_component = noise_component * gain

    return TimelineAudio(
        audio=mixed.astype(np.float32),
        reference=reference,
        noise_component=noise_component,
        gt_segments=segments,
        snr_db_construction=snr_construction,
        peak_limited=peak_limited,
        gain_a=gain_a,
        gain_b=gain_b,
    )


def build_clip_audio(
    speech: np.ndarray,
    noise: np.ndarray | None = None,
    rir: np.ndarray | None = None,
    snr_db: float | None = None,
    target_p99_peak: float | None = TARGET_P99_PEAK,
    margin_samples: int = 8000,
) -> TimelineAudio:
    """Render one whole clip (for hand labelling) with margin_samples of silence
    on each side. The SNR speech reference is the whole clip; the level and
    noise rules are those of build_timeline_audio."""
    if (noise is None) != (snr_db is None):
        raise ValueError("noise and snr_db must be given together")
    gain = 1.0
    if target_p99_peak is not None:
        speech, gain = normalize_level(speech, target_p99_peak)
    start = margin_samples
    end = start + len(speech)
    timeline = np.zeros(end + margin_samples, dtype=np.float32)
    timeline[start:end] = speech
    return _render(timeline, [(start, end)], noise, rir, snr_db, gain, 1.0)
