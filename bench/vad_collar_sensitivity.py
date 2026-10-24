"""Sensitivity of the VAD results to the collar, without hand labels.

The collar enters the scorer three times: a VAD onset further than the collar from
the ground-truth onset is a missed onset, the delay beyond the collar is clipped
speech, and a VAD segment starting outside the collar and outside the ground truth
is a false trigger. This module scores the same VAD output at several collars, so a
conclusion can be stated as "holds for any collar >= X ms" instead of resting on a
provisional collar. It does not estimate the true collar; only Stage 3B can.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from bench.vad_energy import (
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_SPEECH_TIMEOUT_MS,
    EnergyVad,
)
from bench.vad_manifest import VadManifest
from bench.vad_runner import (
    DEFAULT_MANIFEST,
    THRESHOLD_GRID,
    TIMEOUT_GRID_MS,
    ItemResult,
    _AudioCache,
    aggregate,
    prepare_item,
    score_prepared,
    select_items,
)

COLLARS_MS = [0, 25, 50, 75, 100, 125, 150, 200]
ONSET_TOLERANCE_MS = 150
LIMIT_MISS = 0.05
LIMIT_FT = 0.1
LIMIT_CLIP_MS = 50.0

CSV_FIELDS = [
    "threshold", "timeout_ms", "collar_ms", "condition", "snr_db", "n_items",
    "missed_onset_rate", "false_trigger_rate_per_min", "clip_mean_ms", "meets_all",
]  # fmt: skip


def run_collar_sensitivity(
    items: list,
    thresholds: list[float],
    timeouts: list[int],
    collars_ms: list[int],
) -> dict[tuple[float, int, int], list[ItemResult]]:
    timing = {"build_s": 0.0, "vad_s": 0.0, "score_s": 0.0}
    cache = _AudioCache()
    prepared = [prepare_item(it, cache, timing) for it in items]
    results: dict[tuple[float, int, int], list[ItemResult]] = {}
    for thr in thresholds:
        for to in timeouts:
            vad = EnergyVad(thr, to)
            detections = [vad.detect_energy(p.energy, p.n_samples) for p in prepared]
            for collar_ms in collars_ms:
                results[(thr, to, collar_ms)] = [
                    score_prepared(p, vad, collar_ms / 1000.0, timing, result=d)
                    for p, d in zip(prepared, detections, strict=True)
                ]
    return results


def meets_all(cell: dict) -> bool:
    ft = cell["false_trigger_rate_per_min"]
    return (
        cell["missed_onset_rate"] <= LIMIT_MISS
        and ft is not None
        and ft <= LIMIT_FT
        and cell["clipped_ms_per_utterance"]["mean"] <= LIMIT_CLIP_MS
    )


def condition_rows(
    results: dict[tuple[float, int, int], list[ItemResult]],
) -> list[dict]:
    rows = []
    for (thr, to, collar_ms), rs in results.items():
        for cell in aggregate(rs)["cells"]:
            rows.append(
                {
                    "threshold": thr,
                    "timeout_ms": to,
                    "collar_ms": collar_ms,
                    "condition": cell["condition"],
                    "snr_db": "clean" if cell["snr_db"] is None else cell["snr_db"],
                    "n_items": cell["n_items"],
                    "missed_onset_rate": cell["missed_onset_rate"],
                    "false_trigger_rate_per_min": cell["false_trigger_rate_per_min"],
                    "clip_mean_ms": cell["clipped_ms_per_utterance"]["mean"],
                    "meets_all": meets_all(cell),
                }
            )
    return rows


def first_collar_meeting(by_collar: dict[int, bool]) -> int | None:
    """Smallest collar (ms) at which the criteria are met and stay met for every
    larger collar in the grid; None if never."""
    ordered = sorted(by_collar)
    for i, c in enumerate(ordered):
        if all(by_collar[k] for k in ordered[i:]):
            return c
    return None


def _label(row: dict) -> str:
    return f"{row['condition']}/{row['snr_db']}"


def print_report(
    rows: list[dict], collars_ms: list[int], focus: tuple[float, int]
) -> None:
    labels = list(dict.fromkeys(_label(r) for r in rows))
    head = f"{'cond/snr':<16}" + "".join(f"{c:>7}" for c in collars_ms)

    def table(title: str, key: str, fmt: str, scale: float = 1.0) -> None:
        print(f"\n{title} at ADR default {focus} (columns: collar ms)")
        print(head)
        for label in labels:
            cells = []
            for c in collars_ms:
                r = next(
                    x for x in rows
                    if _label(x) == label and x["collar_ms"] == c
                    and (x["threshold"], x["timeout_ms"]) == focus
                )  # fmt: skip
                v = r[key]
                cells.append(
                    "n/a".rjust(7) if v is None else format(v * scale, fmt).rjust(7)
                )
            print(f"{label:<16}" + "".join(cells))

    table("missed-onset %", "missed_onset_rate", ".1f", 100.0)
    table("false triggers per min", "false_trigger_rate_per_min", ".1f")
    table("clipping ms per utterance (mean)", "clip_mean_ms", ".0f")

    configs = sorted({(r["threshold"], r["timeout_ms"]) for r in rows})
    print("\nSmallest collar (ms) from which every criterion is met and stays met")
    print(f"{'cond/snr':<16}{'at ADR default':>16}{'best of 25 configs':>20}")
    for label in labels:
        at_focus = {
            r["collar_ms"]: r["meets_all"]
            for r in rows
            if _label(r) == label and (r["threshold"], r["timeout_ms"]) == focus
        }
        any_cfg = {
            c: any(
                r["meets_all"]
                for r in rows
                if _label(r) == label and r["collar_ms"] == c
            )
            for c in collars_ms
        }
        a, b = first_collar_meeting(at_focus), first_collar_meeting(any_cfg)
        print(
            f"{label:<16}{'never' if a is None else a:>16}{'never' if b is None else b:>20}"
        )
    print(f"(collars above {ONSET_TOLERANCE_MS} ms exceed the Stage 1 onset tolerance)")

    print(
        f"\nConfigurations (of {len(configs)}) meeting every criterion (columns: collar ms)"
    )
    print(head)
    for label in labels:
        counts = []
        for c in collars_ms:
            n = sum(
                1
                for r in rows
                if _label(r) == label and r["collar_ms"] == c and r["meets_all"]
            )
            counts.append(f"{n:>7}")
        print(f"{label:<16}" + "".join(counts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--per-cell-per-lang", type=int, default=None)
    ap.add_argument("--thresholds", default=",".join(str(t) for t in THRESHOLD_GRID))
    ap.add_argument("--timeouts", default=",".join(str(t) for t in TIMEOUT_GRID_MS))
    ap.add_argument("--collars-ms", default=",".join(str(c) for c in COLLARS_MS))
    ap.add_argument("--out", default="bench-results/vad-collar-sensitivity")
    args = ap.parse_args(argv)

    thresholds = [float(x) for x in args.thresholds.split(",")]
    timeouts = [int(x) for x in args.timeouts.split(",")]
    collars_ms = sorted(int(x) for x in args.collars_ms.split(","))
    focus = (DEFAULT_ENERGY_THRESHOLD, DEFAULT_SPEECH_TIMEOUT_MS)
    if focus[0] not in thresholds or focus[1] not in timeouts:
        focus = (thresholds[0], timeouts[0])

    manifest = VadManifest.from_json(args.manifest)
    items = select_items(manifest.items, args.per_cell_per_lang)
    results = run_collar_sensitivity(items, thresholds, timeouts, collars_ms)
    rows = condition_rows(results)

    print(f"collar sensitivity: {len(thresholds)} x {len(timeouts)} configs, "
          f"{len(collars_ms)} collars, items={len(items)}")  # fmt: skip
    print_report(rows, collars_ms, focus)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sensitivity.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (out / "sensitivity.json").write_text(
        json.dumps(
            {"collars_ms": collars_ms, "focus": list(focus), "rows": rows}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out / 'sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
