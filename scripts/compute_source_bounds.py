"""Compute reference speech boundaries for every materialised FLEURS clip.

Run: make vad-bounds   (needs torch + silero-vad; writes bench/vad_source_bounds.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench.vad_noise import TARGET_P99_PEAK, load_mono, normalize_level
from bench.vad_reference import (
    REFERENCE_VAD,
    SOURCE_BOUNDS_PATH,
    silero_bounds,
    silero_defaults,
    silero_version,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audio-root", default="eval_data/vad_audio/fleurs")
    ap.add_argument("--out", default=SOURCE_BOUNDS_PATH)
    args = ap.parse_args(argv)

    wavs = sorted(Path(args.audio_root).glob("*/*.wav"))
    if not wavs:
        print(f"no WAVs under {args.audio_root}", file=sys.stderr)
        return 1

    files: dict[str, dict] = {}
    failed: list[str] = []
    for wav in wavs:
        audio = load_mono(str(wav))
        try:
            normalised, _ = normalize_level(audio)
        except ValueError as exc:
            failed.append(f"{wav.as_posix()}: {exc}")
            continue
        bounds = silero_bounds(normalised)
        if bounds is None:
            failed.append(f"{wav.as_posix()}: Silero found no speech")
            continue
        files[wav.as_posix()] = {
            "onset": bounds[0],
            "offset": bounds[1],
            "n_samples": len(audio),
        }
        print(f"{wav.as_posix()}  {bounds[0] / 16000:.2f}s .. {bounds[1] / 16000:.2f}s")

    if failed:
        print("\nFAILED clips (no bounds written):", file=sys.stderr)
        for line in failed:
            print(f"  {line}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "reference_vad": REFERENCE_VAD,
                "version": silero_version(),
                "params": silero_defaults(),
                "level_target_p99_peak": TARGET_P99_PEAK,
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out} ({len(files)} clips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
