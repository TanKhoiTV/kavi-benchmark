"""Measure intra-utterance pause distributions to justify the VAD silence timeout.

`vad_speech_timeout_ms` (500 ms today) is the trailing silence required before
endpointing fires, so it sits inside the 2.0 s turnaround budget. Cutting it to
300 ms returns 200 ms for no model change, but only if speech does not pause for
300 ms mid-utterance. This script measures that directly on the organizer sample
corpus: silero-vad segments each clip, and the gaps between consecutive speech
segments are the pauses an endpointing timeout would have to survive.

Pauses shorter than `--min-silence-ms` are not separable from normal inter-word
gaps, so that value is the measurement floor and is reported with the results.

Usage:
    uv run python bench/vad_pause_stats.py \
        --set "Vie-Eng=$HOME/Downloads/2026-10-24/Vie-Eng dataset/Vie-Eng/wav" \
        --set "Eng-Vie=$HOME/Downloads/2026-10-24/Vie-Eng dataset/Eng-Vie/wav"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

DEFAULT_OUT = "bench-results/vad-pause-stats"
SAMPLE_RATE = 16_000
# Timeouts whose safety we want to judge.
THRESHOLDS_MS = (200, 250, 300, 400, 500)


def resolve_clips(target: str) -> list[Path]:
    """Accept a directory or a glob and return the wav files it names."""
    path = Path(target).expanduser()
    if path.is_dir():
        return sorted(path.glob("*.wav"))
    return sorted(Path().glob(target)) if any(c in target for c in "*?[") else []


def pause_stats(
    label: str, clips: list[Path], min_silence_ms: int
) -> dict[str, object]:
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    rows: list[dict[str, object]] = []
    gaps: list[float] = []
    clips_with_pauses = 0
    for clip in clips:
        # soundfile rather than silero's read_audio, which now routes through
        # torchcodec under torchaudio 2.11 and needs an extra native dependency.
        data, sr = sf.read(str(clip), dtype="float32", always_2d=True)
        mono = torch.from_numpy(data.mean(axis=1))
        if sr != SAMPLE_RATE:
            import torchaudio.functional as af

            mono = af.resample(mono, orig_freq=sr, new_freq=SAMPLE_RATE)
        audio = mono
        stamps = get_speech_timestamps(
            audio,
            model,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=0,
            threshold=0.5,
        )
        gaps_ms = [
            round((stamps[i + 1]["start"] - stamps[i]["end"]) * 1000 / SAMPLE_RATE, 1)
            for i in range(len(stamps) - 1)
        ]
        gaps.extend(gaps_ms)
        if gaps_ms:
            clips_with_pauses += 1
        rows.append(
            {
                "clip": clip.name,
                "duration_s": round(audio.shape[-1] / SAMPLE_RATE, 3),
                "segments": len(stamps),
                "gaps_ms": gaps_ms,
            }
        )
        model.reset_states()

    sorted_gaps = sorted(gaps)

    def pct(p: float) -> float | None:
        if not sorted_gaps:
            return None
        return sorted_gaps[
            min(len(sorted_gaps) - 1, int(round(p / 100 * (len(sorted_gaps) - 1))))
        ]

    summary: dict[str, object] = {
        "label": label,
        "clips": len(rows),
        "clips_with_pauses": clips_with_pauses,
        "pauses": len(gaps),
        "min_silence_ms": min_silence_ms,
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "max": max(gaps) if gaps else None,
        "mean": round(statistics.fmean(gaps), 1) if gaps else None,
    }
    # How often a timeout would have fired mid-utterance.
    for threshold in THRESHOLDS_MS:
        count = sum(1 for gap in gaps if gap >= threshold)
        summary[f"ge_{threshold}ms"] = count
        summary[f"pct_ge_{threshold}ms"] = (
            round(100 * count / len(gaps), 1) if gaps else None
        )
    return {"summary": summary, "clips": rows}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--set",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="a labelled corpus: directory of wavs, or a glob",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=100,
        help="VAD segmentation floor; gaps below this are not separable from inter-word gaps",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for spec in args.set:
        label, _, target = spec.partition("=")
        if not label or not target:
            raise SystemExit(f"--set expects LABEL=PATH, got {spec!r}")
        clips = resolve_clips(target)
        if not clips:
            raise SystemExit(f"no wav files found for {label} at {target!r}")
        print(f"{label}: {len(clips)} clips", flush=True)
        results[label] = pause_stats(label, clips, args.min_silence_ms)
        (out_dir / f"{label}.json").write_text(
            json.dumps(results[label], indent=2), encoding="utf-8"
        )

    header = ["metric"] + [spec.partition("=")[0] for spec in args.set]
    print()
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")
    keys = [
        "clips",
        "clips_with_pauses",
        "pauses",
        "p50",
        "p95",
        "p99",
        "max",
        "mean",
        *[f"ge_{t}ms" for t in THRESHOLDS_MS],
        *[f"pct_ge_{t}ms" for t in THRESHOLDS_MS],
    ]
    for key in keys:
        row = [key] + [
            str(results[spec.partition("=")[0]]["summary"].get(key))
            for spec in args.set
        ]
        print("| " + " | ".join(row) + " |")

    (out_dir / "summary.json").write_text(
        json.dumps({k: v["summary"] for k, v in results.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
