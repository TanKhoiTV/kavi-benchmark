"""Stage 3B: reference-VAD boundary error against hand-labelled gold, and the collar.

Error is reference minus hand label, in seconds, for onset and offset separately.
The collar is the 95th percentile (nearest rank, no interpolation) of the absolute
errors pooled over onsets and offsets. No other rule is used to pick a collar.
The result is written to bench/vad_collar.json, which bench.vad_runner reads.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from bench.vad_noise import SR, load_mono

COLLAR_PATH = "bench/vad_collar.json"
GOLD_TEMPLATE = "eval_data/vad_gold_template.csv"
GOLD_AUDIO_DIR = "eval_data/vad_gold_audio"
GOLD_LABELS = "eval_data/vad_gold_labels.csv"
PERCENTILE = 95.0
EXPECTED_CLIPS = 50
STAGE1_ONSET_TOLERANCE_S = 0.150
STAGE1_CLIPPING_LIMIT_S = 0.050

Bounds = tuple[float, float]


def nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return float(ordered[rank - 1])


def boundary_errors(
    reference: dict[str, Bounds], hand: dict[str, Bounds]
) -> dict[str, list[float]]:
    ids = sorted(set(reference) & set(hand))
    return {
        "ids": ids,
        "onset": [reference[i][0] - hand[i][0] for i in ids],
        "offset": [reference[i][1] - hand[i][1] for i in ids],
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def summarize(errors: dict, percentile: float = PERCENTILE) -> dict:
    out: dict = {}
    pooled: list[float] = []
    for kind in ("onset", "offset"):
        signed = errors[kind]
        absolute = [abs(e) for e in signed]
        pooled += absolute
        out[kind] = {
            "signed": _stats(signed),
            "absolute": _stats(absolute),
            f"abs_p{percentile:g}": nearest_rank(absolute, percentile)
            if absolute
            else None,
        }
    out["pooled_abs"] = {
        **_stats(pooled),
        f"p{percentile:g}": nearest_rank(pooled, percentile) if pooled else None,
    }
    return out


def derive_collar(errors: dict, percentile: float = PERCENTILE) -> float:
    pooled = [abs(e) for e in errors["onset"]] + [abs(e) for e in errors["offset"]]
    return nearest_rank(pooled, percentile)


def stage1_comparison(collar_s: float) -> dict:
    return {
        "onset_tolerance_s": STAGE1_ONSET_TOLERANCE_S,
        "clipping_limit_s": STAGE1_CLIPPING_LIMIT_S,
        "collar_exceeds_onset_tolerance": collar_s > STAGE1_ONSET_TOLERANCE_S,
        "collar_exceeds_clipping_limit": collar_s > STAGE1_CLIPPING_LIMIT_S,
    }


def load_labels(path: str) -> tuple[dict[str, Bounds], int]:
    labels: dict[str, Bounds] = {}
    blank = 0
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            on, off = (
                (row.get("onset_s") or "").strip(),
                (row.get("offset_s") or "").strip(),
            )
            if not on and not off:
                blank += 1
                continue
            onset, offset = float(on), float(off)
            if not 0 <= onset < offset:
                raise ValueError(f"{path}: invalid label for {row['id']}: {on}, {off}")
            labels[row["id"]] = (onset, offset)
    return labels, blank


def load_conditions(path: str) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8", newline="") as f:
        return {r["id"]: f"{r['condition']}/{r['snr_db']}" for r in csv.DictReader(f)}


def reference_bounds(ids: list[str], audio_dir: str) -> dict[str, Bounds]:
    from bench.vad_reference import silero_bounds

    out: dict[str, Bounds] = {}
    for clip_id in ids:
        wav = Path(audio_dir) / f"{clip_id}.wav"
        if not wav.exists():
            raise FileNotFoundError(f"{wav} missing; run `make vad-gold-audio`")
        found = silero_bounds(load_mono(str(wav)))
        if found is None:
            raise RuntimeError(f"{clip_id}: reference VAD found no speech")
        out[clip_id] = (found[0] / SR, found[1] / SR)
    return out


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 1000:+.0f}"


def print_report(summary: dict, collar_s: float, findings: dict, n_clips: int) -> None:
    print(f"gold clips: {n_clips} | boundaries: {2 * n_clips}")
    print("error = reference - hand label (ms); abs stats use |error|")
    print(f"{'':<8}{'mean':>8}{'median':>8}{'absMean':>9}{'absMed':>8}{'absP95':>8}")
    for kind in ("onset", "offset"):
        s = summary[kind]
        print(
            f"{kind:<8}{_ms(s['signed']['mean']):>8}{_ms(s['signed']['median']):>8}"
            f"{_ms(s['absolute']['mean']):>9}{_ms(s['absolute']['median']):>8}"
            f"{_ms(s[f'abs_p{PERCENTILE:g}']):>8}"
        )
    print(
        f"collar (pooled |error| p{PERCENTILE:g}, nearest rank): {collar_s * 1000:.0f} ms"
    )
    print(
        f"Stage 1 onset tolerance {findings['onset_tolerance_s'] * 1000:.0f} ms | "
        f"clipped-speech limit {findings['clipping_limit_s'] * 1000:.0f} ms"
    )
    if findings["collar_exceeds_onset_tolerance"]:
        print("WARNING: collar exceeds the Stage 1 onset tolerance")
    if findings["collar_exceeds_clipping_limit"]:
        print("WARNING: collar exceeds the Stage 1 clipped-speech limit")
    if not (
        findings["collar_exceeds_onset_tolerance"]
        or findings["collar_exceeds_clipping_limit"]
    ):
        print("collar is within both Stage 1 limits")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=GOLD_LABELS)
    ap.add_argument("--audio-dir", default=GOLD_AUDIO_DIR)
    ap.add_argument("--gold-template", default=GOLD_TEMPLATE)
    ap.add_argument("--out", default=COLLAR_PATH)
    ap.add_argument("--expect-clips", type=int, default=EXPECTED_CLIPS)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args(argv)

    hand, blank = load_labels(args.labels)
    if not hand:
        print(
            f"no labelled clips in {args.labels} ({blank} blank rows); "
            "fill onset_s and offset_s for at least one clip first",
            file=sys.stderr,
        )
        return 1
    if len(hand) != args.expect_clips and not args.allow_partial:
        print(
            f"{len(hand)} labelled clips ({blank} blank), expected {args.expect_clips}; "
            "finish labelling or pass --allow-partial (result is then not a Stage 3B collar)",
            file=sys.stderr,
        )
        return 1

    reference = reference_bounds(sorted(hand), args.audio_dir)
    errors = boundary_errors(reference, hand)
    summary = summarize(errors)
    collar_s = derive_collar(errors)
    findings = stage1_comparison(collar_s)

    conditions = load_conditions(args.gold_template)
    by_condition: dict[str, list[float]] = {}
    for i, clip_id in enumerate(errors["ids"]):
        key = conditions.get(clip_id, "unknown")
        by_condition.setdefault(key, []).extend(
            [abs(errors["onset"][i]), abs(errors["offset"][i])]
        )

    print_report(summary, collar_s, findings, len(errors["ids"]))
    print("per condition/SNR |error| p95 (ms):")
    for key in sorted(by_condition):
        print(f"  {key:<16}{nearest_rank(by_condition[key], PERCENTILE) * 1000:>6.0f}")

    from bench.vad_reference import REFERENCE_VAD, silero_defaults, silero_version

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "collar_s": collar_s,
                "rule": f"p{PERCENTILE:g} nearest rank of pooled |reference - hand| "
                "over onsets and offsets",
                "percentile": PERCENTILE,
                "n_clips": len(errors["ids"]),
                "partial": len(errors["ids"]) != args.expect_clips,
                "reference_vad": REFERENCE_VAD,
                "reference_vad_version": silero_version(),
                "reference_vad_params": silero_defaults(),
                "error_unit": "seconds, reference minus hand",
                "statistics": summary,
                "by_condition_abs_p95_s": {
                    k: nearest_rank(v, PERCENTILE)
                    for k, v in sorted(by_condition.items())
                },
                "stage1": findings,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
