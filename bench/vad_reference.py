"""Reference speech-boundary VAD (Silero) and the committed source-bounds file.

The benchmark ground truth is the span of real speech inside each FLEURS clip,
not the whole clip. Those spans come from Silero VAD run with the library's
default parameters on the level-normalised clip. They are computed once by
scripts/compute_source_bounds.py and committed to bench/vad_source_bounds.json so
that manifest generation stays deterministic and needs neither torch nor Silero.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

SOURCE_BOUNDS_PATH = "bench/vad_source_bounds.json"
REFERENCE_VAD = "silero-vad"
SAMPLE_RATE = 16000


def outer_bounds(timestamps: list[dict]) -> tuple[int, int] | None:
    if not timestamps:
        return None
    return (
        int(min(t["start"] for t in timestamps)),
        int(max(t["end"] for t in timestamps)),
    )


@lru_cache(maxsize=1)
def _silero_model():
    from silero_vad import load_silero_vad

    return load_silero_vad()


def silero_defaults() -> dict:
    import inspect

    from silero_vad import get_speech_timestamps

    params = inspect.signature(get_speech_timestamps).parameters
    keys = (
        "threshold",
        "min_speech_duration_ms",
        "min_silence_duration_ms",
        "speech_pad_ms",
    )
    return {k: params[k].default for k in keys if k in params}


def silero_version() -> str:
    from importlib.metadata import version

    return version("silero-vad")


def silero_bounds(audio: np.ndarray) -> tuple[int, int] | None:
    import torch
    from silero_vad import get_speech_timestamps

    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    stamps = get_speech_timestamps(wav, _silero_model(), sampling_rate=SAMPLE_RATE)
    return outer_bounds(stamps)


def load_source_bounds(path: str = SOURCE_BOUNDS_PATH) -> dict[str, tuple[int, int]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found; run `make vad-bounds` and commit the result"
        )
    files = json.loads(p.read_text(encoding="utf-8")).get("files", {})
    out: dict[str, tuple[int, int]] = {}
    for name, entry in files.items():
        onset, offset, n = entry["onset"], entry["offset"], entry["n_samples"]
        if not 0 <= onset < offset <= n:
            raise ValueError(f"{path}: invalid bounds for {name}: {entry}")
        out[name] = (int(onset), int(offset))
    return out
