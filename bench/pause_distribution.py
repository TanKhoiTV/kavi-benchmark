"""Intra-utterance pause distributions, to justify `vad_speech_timeout_ms`.

`vad_speech_timeout_ms` is the trailing silence a VAD must observe before it
declares the utterance finished. Any pause inside an utterance that reaches that
value truncates the sentence, so the decision variable is the *largest* pause the
corpus contains, not its mean.

This measures those pauses directly: Silero VAD segments the clip, and every gap
between consecutive speech segments - excluding the leading and trailing silence,
which are not pauses - is one sample. Pause durations are quantised to Silero's
512-sample frame (32 ms at 16 kHz), so a timeout must clear the observed maximum by
at least one frame.

Two choices are deliberately conservative:
  * `speech_pad_ms=0` (the library default of 30 ms would pad segments and shrink
    every measured pause);
  * `min_silence_duration_ms` is one frame, so short silences are not merged away
    before they can be counted.

Threshold sensitivity is reported rather than assumed: a stricter detector drops
more marginal speech and therefore produces longer pauses, which is the direction
that makes a shorter timeout look unsafe.

Usage:
    python -m bench.pause_distribution \
        --wav-dir VI=/path/to/wavs --wav-dir EN=/path/to/wavs \
        --out bench-results/pause-distribution.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.vad_noise import SR, load_mono
from bench.vad_reference import _silero_model

FRAME = 512  # Silero's analysis window at 16 kHz
FRAME_MS = FRAME / SR * 1000.0  # 32 ms

# Candidate timeouts, and the long-gap cut used to separate the inter-word gaps
# that must never end an utterance from the inter-sentence pauses that arguably may.
CANDIDATE_TIMEOUTS_MS = (200, 250, 300, 350, 400, 450, 500)
SENTENCE_PAUSE_MS = 700.0


def speech_segments(audio: np.ndarray, threshold: float) -> list[dict]:
    import torch
    from silero_vad import get_speech_timestamps

    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    return get_speech_timestamps(
        wav,
        _silero_model(),
        sampling_rate=SR,
        threshold=threshold,
        min_speech_duration_ms=int(FRAME_MS),
        min_silence_duration_ms=int(FRAME_MS),
        speech_pad_ms=0,
    )


def pauses_ms(stamps: list[dict]) -> list[float]:
    """Gaps between consecutive speech segments, in milliseconds."""
    return [
        (later["start"] - earlier["end"]) / SR * 1000.0
        # Deliberately staggered: equal length would consume the whole list.
        for earlier, later in zip(stamps, stamps[1:], strict=False)
    ]


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, no interpolation (matches the VAD docs in this repo)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int(np.ceil(p / 100.0 * len(ordered))))
    return float(ordered[rank - 1])


def summarise(pauses: list[float], clips: int) -> dict:
    return {
        "clips": clips,
        "pauses": len(pauses),
        "p50_ms": round(percentile(pauses, 50), 1),
        "p90_ms": round(percentile(pauses, 90), 1),
        "p95_ms": round(percentile(pauses, 95), 1),
        "p99_ms": round(percentile(pauses, 99), 1),
        "max_ms": round(max(pauses), 1) if pauses else 0.0,
        "over_word": sum(1 for p in pauses if p >= SENTENCE_PAUSE_MS),
        "reaching": {
            str(t): sum(1 for p in pauses if p >= t) for t in CANDIDATE_TIMEOUTS_MS
        },
    }


def measure(wav_dir: Path, threshold: float, limit: int | None = None) -> dict:
    files = sorted(wav_dir.glob("*.wav"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no .wav files under {wav_dir}")

    all_pauses: list[float] = []
    per_clip: dict[str, list[float]] = {}
    durations: list[float] = []
    for path in files:
        audio = load_mono(str(path))
        durations.append(len(audio) / SR)
        gaps = pauses_ms(speech_segments(audio, threshold))
        per_clip[path.stem] = gaps
        all_pauses.extend(gaps)

    word_pauses = [p for p in all_pauses if p < SENTENCE_PAUSE_MS]
    stats = summarise(all_pauses, len(files))
    # A single-segment clip has no pauses at all, so it cannot carry the maximum.
    with_pauses = {k: v for k, v in per_clip.items() if v}
    stats["max_pause_clip"] = (
        max(with_pauses, key=lambda k: max(with_pauses[k])) if with_pauses else None
    )
    stats["truncated_clips"] = {
        str(t): sum(1 for gaps in per_clip.values() if any(g >= t for g in gaps))
        for t in CANDIDATE_TIMEOUTS_MS
    }
    stats["inter_word"] = summarise(word_pauses, len(files))
    stats["per_clip"] = {k: [round(v, 1) for v in vals] for k, vals in per_clip.items()}
    stats["mean_clip_s"] = round(float(np.mean(durations)), 2)
    stats["threshold"] = threshold
    stats["frame_ms"] = FRAME_MS
    return stats


def table(label: str, stats: dict) -> str:
    lines = [
        f"### {label} (Silero threshold {stats['threshold']}, "
        f"{stats['clips']} clips, mean {stats['mean_clip_s']} s)",
        "",
        "| set | pauses | p50 | p90 | p95 | p99 | max |",
        "|---|---|---|---|---|---|---|",
        (
            f"| all in-speech gaps | {stats['pauses']} | {stats['p50_ms']} | "
            f"{stats['p90_ms']} | {stats['p95_ms']} | {stats['p99_ms']} | "
            f"**{stats['max_ms']}** |"
        ),
        (
            f"| gaps < {SENTENCE_PAUSE_MS:.0f} ms | {stats['inter_word']['pauses']} | "
            f"{stats['inter_word']['p50_ms']} | {stats['inter_word']['p90_ms']} | "
            f"{stats['inter_word']['p95_ms']} | {stats['inter_word']['p99_ms']} | "
            f"**{stats['inter_word']['max_ms']}** |"
        ),
        "",
        "| candidate timeout | pauses reaching it | clips truncated |",
        "|---|---|---|",
    ]
    for t in CANDIDATE_TIMEOUTS_MS:
        lines.append(
            f"| {t} ms | {stats['reaching'][str(t)]} | {stats['truncated_clips'][str(t)]} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wav-dir",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="corpus label and directory of 16 kHz mono wav files (repeatable)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.7],
        help="Silero speech thresholds to report (0.7 is the conservative direction)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="use only the first N clips"
    )
    parser.add_argument("--out", default="bench-results/pause-distribution.json")
    args = parser.parse_args(argv)

    corpora: list[tuple[str, Path]] = []
    for spec in args.wav_dir:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--wav-dir needs LABEL=PATH, got {spec!r}")
        corpora.append((label, Path(path)))

    results: dict = {
        "frame_ms": FRAME_MS,
        "sentence_pause_ms": SENTENCE_PAUSE_MS,
        "corpora": {},
    }
    for label, path in corpora:
        for threshold in args.thresholds:
            stats = measure(path, threshold, args.limit)
            results["corpora"][f"{label}@{threshold}"] = stats
            print(table(f"{label} @ threshold {threshold}", stats))
            print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
