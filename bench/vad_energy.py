"""Energy-based VAD candidate for ADR-022 (frame peak amplitude, single threshold).

ADR-022 fixes only: an amplitude threshold on the PCM float stream (line 33) and
a configurable silence period, default ~500 ms, that finalises an utterance
(line 34). Everything else below is a gap-fill, recorded as an assumption:

- Frame: 10 ms (160 samples at 16 kHz), non-overlapping. Every Stage 1 pause and
  timeout value is a whole number of 10 ms frames.
- Energy: peak absolute sample value of the frame, compared on the float scale.
- Speech frame: energy >= energy_threshold.
- Hysteresis: none; one threshold for both starting and ending speech.
- Hang-in: 1 frame; the first speech frame opens a segment.
- Hang-out: a segment closes when the run of non-speech frames is strictly
  longer than speech_timeout_ms, so a pause equal to the timeout never splits.
- Segment offset: end of the last speech frame. The end of speech is declared one
  frame after the timeout has been exceeded (eos_s); it is None when the audio
  ends before the timeout is exceeded.
- No pre-roll, no maximum length, no smoothing, no adaptive threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 10
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
DEFAULT_ENERGY_THRESHOLD = 0.05
DEFAULT_SPEECH_TIMEOUT_MS = 500


@dataclass(frozen=True)
class EnergyVadResult:
    segments: list[tuple[float, float]]
    eos_s: list[float | None]


class EnergyVad:
    def __init__(
        self,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
        speech_timeout_ms: int = DEFAULT_SPEECH_TIMEOUT_MS,
    ) -> None:
        if energy_threshold <= 0:
            raise ValueError("energy_threshold must be positive")
        if speech_timeout_ms <= 0 or speech_timeout_ms % FRAME_MS != 0:
            raise ValueError(
                f"speech_timeout_ms must be a positive multiple of {FRAME_MS}"
            )
        self.energy_threshold = float(energy_threshold)
        self.speech_timeout_ms = int(speech_timeout_ms)
        self.timeout_frames = self.speech_timeout_ms // FRAME_MS

    def frame_energy(self, audio: np.ndarray) -> np.ndarray:
        x = np.abs(np.asarray(audio, dtype=np.float64))
        n_frames = -(-len(x) // FRAME_SAMPLES)
        padded = np.zeros(n_frames * FRAME_SAMPLES, dtype=np.float64)
        padded[: len(x)] = x
        return padded.reshape(n_frames, FRAME_SAMPLES).max(axis=1)

    def detect(self, audio: np.ndarray) -> EnergyVadResult:
        return self.detect_energy(self.frame_energy(audio), len(audio))

    def detect_energy(self, energy: np.ndarray, n_samples: int) -> EnergyVadResult:
        """Run the state machine on precomputed frame_energy(); identical to detect()."""
        is_speech = energy >= self.energy_threshold
        n_audio_s = n_samples / SAMPLE_RATE
        frame_s = FRAME_MS / 1000.0

        segments: list[tuple[float, float]] = []
        eos_s: list[float | None] = []
        start: int | None = None
        last_speech = -1
        silence_run = 0

        for i, speech in enumerate(is_speech):
            if start is None:
                if speech:
                    start = i
                    last_speech = i
                    silence_run = 0
                continue
            if speech:
                last_speech = i
                silence_run = 0
                continue
            silence_run += 1
            if silence_run > self.timeout_frames:
                offset = min((last_speech + 1) * frame_s, n_audio_s)
                segments.append((start * frame_s, offset))
                eos_s.append(offset + (self.timeout_frames + 1) * frame_s)
                start = None

        if start is not None:
            offset = min((last_speech + 1) * frame_s, n_audio_s)
            segments.append((start * frame_s, offset))
            eos_s.append(None)

        return EnergyVadResult(segments=segments, eos_s=eos_s)
