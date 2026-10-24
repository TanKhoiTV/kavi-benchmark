"""Threshold x timeout sweep on the VAD manifest (Stage 1 grid), VAD-only.

Each item's audio is built once; only its 10 ms frame energies are kept, then every
(energy_threshold, speech_timeout_ms) pair is scored with the same code path as
bench.vad_runner. ADR defaults are not changed: they are just one grid point.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from bench.vad_borderline import assess, false_trigger_upper95
from bench.vad_energy import (
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_SPEECH_TIMEOUT_MS,
    EnergyVad,
)
from bench.vad_manifest import VadManifest
from bench.vad_runner import (
    COLLAR_PATH,
    DEFAULT_MANIFEST,
    THRESHOLD_GRID,
    TIMEOUT_GRID_MS,
    ItemResult,
    _AudioCache,
    aggregate,
    load_collar,
    prepare_item,
    score_prepared,
    select_items,
    summarize,
)

METRICS = (
    "missed_onset_rate",
    "false_trigger_rate_per_min",
    "clipped_ms_per_utterance",
)


def run_sweep(
    items: list,
    thresholds: list[float],
    timeouts: list[int],
    collar_s: float,
) -> tuple[dict[tuple[float, int], list[ItemResult]], dict[str, float]]:
    timing = {"build_s": 0.0, "vad_s": 0.0, "score_s": 0.0}
    t0 = time.perf_counter()
    cache = _AudioCache()
    prepared = [prepare_item(it, cache, timing) for it in items]
    results: dict[tuple[float, int], list[ItemResult]] = {}
    for thr in thresholds:
        for to in timeouts:
            vad = EnergyVad(thr, to)
            results[(thr, to)] = [
                score_prepared(p, vad, collar_s, timing) for p in prepared
            ]
    timing["wall_s"] = time.perf_counter() - t0
    return results, timing


def cell_assessments(results: list[ItemResult]) -> dict:
    """Assess each condition x SNR x pause cell against the hard criteria."""
    groups: dict[tuple, list[ItemResult]] = defaultdict(list)
    for r in results:
        groups[(r.condition, r.snr_db, r.pause_ms)].append(r)
    counts = {
        m: {"pass": 0, "fail": 0, "borderline": 0, "undetermined": 0} for m in METRICS
    }
    sizes = set()
    for rs in groups.values():
        s = summarize(rs)
        sizes.add(len(rs) // 2)
        values = {
            "missed_onset_rate": s["missed_onset_rate"],
            "false_trigger_rate_per_min": s["false_trigger_rate_per_min"],
            "clipped_ms_per_utterance": s["clipped_ms_per_utterance"]["mean"],
        }
        for name, value in values.items():
            counts[name][assess(name, value)] += 1
    return {
        "n_cells": len(groups),
        "items_per_lang_per_cell": sorted(sizes),
        "counts": counts,
    }


def summarize_config(results: list[ItemResult]) -> dict:
    agg = aggregate(results)
    overall = agg["overall"]
    non_speech_min = sum(r.non_speech_s for r in results) / 60.0
    agg["criteria"] = cell_assessments(results)
    agg["false_trigger_upper95_per_min"] = false_trigger_upper95(
        overall["false_trigger_count"], non_speech_min
    )
    agg["non_speech_minutes"] = non_speech_min
    return agg


CSV_FIELDS = [
    "threshold", "timeout_ms", "condition", "snr_db", "n_items",
    "false_trigger_rate_per_min", "missed_onset_rate", "clip_mean_ms", "clip_p95_ms",
    "clip_max_ms", "junction_split_p200", "junction_split_p350", "junction_split_p500",
    "junction_split_p700", "junction_violations", "utterance_split_rate",
    "eou_median_ms", "eou_p95_ms", "eos_median_ms", "eos_undeclared",
]  # fmt: skip


def csv_rows(thr: float, to: int, summary: dict) -> list[dict]:
    rows = []
    for cell in [
        *summary["cells"],
        {**summary["overall"], "condition": "ALL", "snr_db": ""},
    ]:
        by_pause = cell["junction_split_rate_by_pause"]
        rows.append(
            {
                "threshold": thr,
                "timeout_ms": to,
                "condition": cell["condition"],
                "snr_db": "clean" if cell["snr_db"] is None else cell["snr_db"],
                "n_items": cell["n_items"],
                "false_trigger_rate_per_min": cell["false_trigger_rate_per_min"],
                "missed_onset_rate": cell["missed_onset_rate"],
                "clip_mean_ms": cell["clipped_ms_per_utterance"]["mean"],
                "clip_p95_ms": cell["clipped_ms_per_utterance"]["p95"],
                "clip_max_ms": cell["clipped_ms_per_utterance"]["max"],
                **{
                    f"junction_split_p{p}": by_pause.get(str(p))
                    for p in (200, 350, 500, 700)
                },
                "junction_violations": cell["junction_violations"],
                "utterance_split_rate": cell["utterance_split_rate"],
                "eou_median_ms": cell["eou_delay_ms"]["median"],
                "eou_p95_ms": cell["eou_delay_ms"]["p95"],
                "eos_median_ms": cell["eos_delay_ms"]["median"],
                "eos_undeclared": cell["eos_undeclared"],
            }
        )
    return rows


def _matrix(
    title: str, values: dict, thresholds: list[float], timeouts: list[int], fmt: str
) -> None:
    print(f"\n{title}  (rows: threshold, columns: speech_timeout_ms)")
    print(f"{'':>8}" + "".join(f"{t:>9}" for t in timeouts))
    for thr in thresholds:
        cells = []
        for to in timeouts:
            v = values[(thr, to)]
            cells.append(f"{'n/a':>9}" if v is None else format(v, fmt).rjust(9))
        print(f"{thr:>8}" + "".join(cells))


def print_matrices(
    summaries: dict, thresholds: list[float], timeouts: list[int]
) -> None:
    def pick(fn):
        return {k: fn(v) for k, v in summaries.items()}

    _matrix("missed-onset rate % (overall)", pick(lambda s: s["overall"]["missed_onset_rate"] * 100),
            thresholds, timeouts, ".1f")  # fmt: skip
    _matrix("false triggers per min (overall)", pick(lambda s: s["overall"]["false_trigger_rate_per_min"]),
            thresholds, timeouts, ".2f")  # fmt: skip
    _matrix("clipping ms per utterance, mean (overall)", pick(lambda s: s["overall"]["clipped_ms_per_utterance"]["mean"]),
            thresholds, timeouts, ".0f")  # fmt: skip
    _matrix("junction violations (A-B pause <= timeout not bridged)", pick(lambda s: s["overall"]["junction_violations"]),
            thresholds, timeouts, ".0f")  # fmt: skip
    _matrix("borderline cells, missed-onset", pick(lambda s: s["criteria"]["counts"]["missed_onset_rate"]["borderline"]),
            thresholds, timeouts, ".0f")  # fmt: skip
    _matrix("cells failing the missed-onset criterion", pick(lambda s: s["criteria"]["counts"]["missed_onset_rate"]["fail"]),
            thresholds, timeouts, ".0f")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--per-cell-per-lang", type=int, default=None)
    ap.add_argument("--thresholds", default=",".join(str(t) for t in THRESHOLD_GRID))
    ap.add_argument("--timeouts", default=",".join(str(t) for t in TIMEOUT_GRID_MS))
    ap.add_argument("--collar-file", default=COLLAR_PATH)
    ap.add_argument("--out", default="bench-results/vad-sweep")
    args = ap.parse_args(argv)

    thresholds = [float(x) for x in args.thresholds.split(",")]
    timeouts = [int(x) for x in args.timeouts.split(",")]
    manifest = VadManifest.from_json(args.manifest)
    items = select_items(manifest.items, args.per_cell_per_lang)
    collar_s, provisional = load_collar(args.collar_file)

    results, timing = run_sweep(items, thresholds, timeouts, collar_s)
    summaries = {k: summarize_config(v) for k, v in results.items()}

    tag = " (PROVISIONAL, Stage 3B not run)" if provisional else ""
    print(
        f"sweep: {len(thresholds)} thresholds x {len(timeouts)} timeouts | items={len(items)}"
    )
    print(f"collar: {collar_s:.3f}s{tag}")
    default = (DEFAULT_ENERGY_THRESHOLD, DEFAULT_SPEECH_TIMEOUT_MS)
    if default in summaries:
        print(
            f"ADR default grid point: energy_threshold={default[0]} speech_timeout_ms={default[1]}"
        )
    print_matrices(summaries, thresholds, timeouts)

    ns = next(iter(summaries.values()))["non_speech_minutes"]
    print(f"\nnon-speech time in this sample: {ns:.1f} min "
          f"(0 false triggers shows <= {3.0 / ns:.2f}/min at 95% confidence; criterion is 0.1/min)")  # fmt: skip
    print(f"runtime: wall={timing['wall_s']:.1f}s build={timing['build_s']:.1f}s "
          f"vad={timing['vad_s']:.1f}s score={timing['score_s']:.1f}s")  # fmt: skip

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(
        json.dumps(
            {
                "manifest": args.manifest,
                "n_items": len(items),
                "collar_s": collar_s,
                "collar_provisional": provisional,
                "timing": timing,
                "configs": [
                    {"threshold": thr, "timeout_ms": to, **summaries[(thr, to)]}
                    for thr in thresholds
                    for to in timeouts
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with open(out / "sweep.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        w.writeheader()
        for (thr, to), s in summaries.items():
            w.writerows(csv_rows(thr, to, s))
    print(f"wrote {out / 'sweep.json'} and {out / 'sweep.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
