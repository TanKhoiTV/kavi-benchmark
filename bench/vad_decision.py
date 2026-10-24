"""Stage 6 helper: apply the criteria mechanically to every (threshold, timeout) pair.

Reads the manifest and the Stage 3B collar, scores each pair at the collar and at the
collar -/+ the label error, and classifies every condition x SNR x pause cell as pass,
fail or borderline per criterion. It never edits repository files; it writes only to
bench-results/vad-decision/.

The criteria and margins are those in bench.vad_borderline (assumptions until the
Stage 1 text is checked). A cell passes only if every criterion is a clean pass; a
borderline value is not a pass. Cells whose status changes between collar - error and
collar + error are flagged as label-error sensitive and are not decided.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.vad_borderline import (
    CRITERIA,
    ITEMS_PER_CELL,
    ITEMS_PER_CELL_WHEN_BORDERLINE,
    assess,
)
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
    prepare_item,
    score_prepared,
)

NAMES = tuple(CRITERIA)
GATE_MS = 2000


def cell_values(rs: list[ItemResult]) -> dict[str, float | None]:
    non_speech_min = sum(r.non_speech_s for r in rs) / 60.0
    ft = sum(r.metrics.false_trigger_count for r in rs)
    return {
        "missed_onset_rate": sum(r.metrics.missed_onset_count for r in rs) / len(rs),
        "false_trigger_rate_per_min": ft / non_speech_min
        if non_speech_min > 0
        else None,
        "clipped_ms_per_utterance": float(
            np.mean([r.metrics.clipped_start_ms + r.metrics.clipped_end_ms for r in rs])
        ),
    }


def cell_statuses(values: dict[str, float | None]) -> dict[str, str]:
    return {name: assess(name, values[name]) for name in NAMES}


def all_pass(statuses: dict[str, str]) -> bool:
    return all(s == "pass" for s in statuses.values())


def has_fail(statuses: dict[str, str]) -> bool:
    return any(s == "fail" for s in statuses.values())


def evaluate_pairs(
    items: list,
    thresholds: list[float],
    timeouts: list[int],
    collar_s: float,
    label_error_s: float,
) -> tuple[list[dict], dict[tuple[float, int], dict]]:
    timing = {"build_s": 0.0, "vad_s": 0.0, "score_s": 0.0}
    cache = _AudioCache()
    prepared = [prepare_item(it, cache, timing) for it in items]
    collars = [max(0.0, collar_s - label_error_s), collar_s, collar_s + label_error_s]
    rows: list[dict] = []
    summaries: dict[tuple[float, int], dict] = {}
    for thr in thresholds:
        for to in timeouts:
            vad = EnergyVad(thr, to)
            detections = [vad.detect_energy(p.energy, p.n_samples) for p in prepared]
            by_collar = []
            for c in collars:
                res = [
                    score_prepared(p, vad, c, timing, result=d)
                    for p, d in zip(prepared, detections, strict=True)
                ]
                groups: dict[tuple, list[ItemResult]] = defaultdict(list)
                for r in res:
                    groups[(r.condition, r.snr_db, r.pause_ms)].append(r)
                by_collar.append(groups)
            eou = [
                r.metrics.end_of_utterance_delay_ms
                for g in [by_collar[1]]
                for rs in g.values()
                for r in rs
            ]
            cells = []
            for key in by_collar[1]:
                mid = cell_values(by_collar[1][key])
                st_mid = cell_statuses(mid)
                st_lo = cell_statuses(cell_values(by_collar[0][key]))
                st_hi = cell_statuses(cell_values(by_collar[2][key]))
                sensitive = st_lo != st_hi
                row = {
                    "threshold": thr, "timeout_ms": to, "condition": key[0],
                    "snr_db": "clean" if key[1] is None else key[1], "pause_ms": key[2],
                    "items": len(by_collar[1][key]), **mid,
                    **{f"status_{n}": st_mid[n] for n in NAMES},
                    "label_error_sensitive": sensitive, "all_pass": all_pass(st_mid),
                }  # fmt: skip
                rows.append(row)
                cells.append(row)
            summaries[(thr, to)] = {
                "cells": len(cells),
                "all_pass_cells": sum(r["all_pass"] for r in cells),
                "failing_cells": sum(has_fail({n: r[f"status_{n}"] for n in NAMES}) for r in cells),
                "borderline_only_cells": sum(
                    not has_fail({n: r[f"status_{n}"] for n in NAMES})
                    and not r["all_pass"]
                    for r in cells
                ),
                "label_sensitive_cells": sum(r["label_error_sensitive"] for r in cells),
                "fails_by_criterion": {
                    n: sum(r[f"status_{n}"] == "fail" for r in cells) for n in NAMES
                },
                "eou_p95_ms": float(np.percentile(eou, 95)) if eou else None,
                "passes_everywhere": all(r["all_pass"] for r in cells),
            }  # fmt: skip
    return rows, summaries


def decide_case(
    summaries: dict[tuple[float, int], dict],
) -> tuple[str, list[tuple[float, int]]]:
    passing = [k for k, s in summaries.items() if s["passes_everywhere"]]
    return ("A" if passing else "B"), passing


def failing_cells_by_condition(
    rows: list[dict], pair: tuple[float, int]
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        if (r["threshold"], r["timeout_ms"]) != pair or r["all_pass"]:
            continue
        key = f"{r['condition']}/{r['snr_db']}"
        entry = out.setdefault(key, {"cells": 0, **{n: 0 for n in NAMES}})
        entry["cells"] += 1
        for n in NAMES:
            entry[n] += r[f"status_{n}"] == "fail"
    return out


def _matrix(title: str, summaries: dict, key: str, thresholds, timeouts) -> None:
    print(f"\n{title}  (rows: threshold, columns: speech_timeout_ms)")
    print(f"{'':>8}" + "".join(f"{t:>8}" for t in timeouts))
    for thr in thresholds:
        print(
            f"{thr:>8}" + "".join(f"{summaries[(thr, to)][key]:>8}" for to in timeouts)
        )


def print_report(rows, summaries, thresholds, timeouts, collar_s, label_error_s) -> str:
    n_cells = next(iter(summaries.values()))["cells"]
    print(
        f"collar {collar_s * 1000:.0f} ms, label error band +/-{label_error_s * 1000:.0f} ms"
    )
    print("criteria (assumed until checked against Stage 1):")
    for name, c in CRITERIA.items():
        print(
            f"  {name}: limit {c.limit} ({c.kind}); borderline margin in vad_borderline.py"
        )
    print(f"pairs: {len(summaries)}, cells per pair: {n_cells}")

    _matrix(
        f"cells with a failing criterion (of {n_cells})",
        summaries,
        "failing_cells",
        thresholds,
        timeouts,
    )
    _matrix(
        "borderline-only cells (no fail, not a clean pass)",
        summaries,
        "borderline_only_cells",
        thresholds,
        timeouts,
    )
    _matrix(
        "cells sensitive to the label error",
        summaries,
        "label_sensitive_cells",
        thresholds,
        timeouts,
    )
    _matrix(
        "cells passing every criterion",
        summaries,
        "all_pass_cells",
        thresholds,
        timeouts,
    )

    case, passing = decide_case(summaries)
    print(f"\nPairs passing in all {n_cells} cells: {passing if passing else 'none'}")
    if case == "B":
        print(
            "CASE B: no (threshold, timeout) pair passes in all cells -> model-based VAD follow-up"
        )
    else:
        print("CASE A candidate pairs (tie-break: apply the confirmed Stage 1 rule):")
        for pair in passing:
            print(f"  {pair}: EOU p95 {summaries[pair]['eou_p95_ms']:.0f} ms")

    fewest = min(summaries, key=lambda k: (summaries[k]["failing_cells"], k))
    focus = [(DEFAULT_ENERGY_THRESHOLD, DEFAULT_SPEECH_TIMEOUT_MS), fewest]
    for pair in dict.fromkeys(focus):
        label = "ADR default" if pair == focus[0] else "fewest failing cells"
        print(
            f"\nFailing cells at {pair} ({label}); counts of cells failing each criterion"
        )
        print(
            f"  {'cond/snr':<16}{'cells':>6}" + "".join(f"{n[:12]:>14}" for n in NAMES)
        )
        for key, e in failing_cells_by_condition(rows, pair).items():
            print(f"  {key:<16}{e['cells']:>6}" + "".join(f"{e[n]:>14}" for n in NAMES))
        s = summaries[pair]
        extra = s["borderline_only_cells"] + s["label_sensitive_cells"]
        print(
            f"  not decided: {s['borderline_only_cells']} borderline-only and "
            f"{s['label_sensitive_cells']} label-error-sensitive cells; at most "
            f"{extra * (ITEMS_PER_CELL_WHEN_BORDERLINE - ITEMS_PER_CELL)} more items "
            f"({ITEMS_PER_CELL_WHEN_BORDERLINE - ITEMS_PER_CELL} per cell, {ITEMS_PER_CELL} -> "
            f"{ITEMS_PER_CELL_WHEN_BORDERLINE})"
        )
        print(f"  timeout {pair[1]} ms; EOU delay p95 {s['eou_p95_ms']:.0f} ms; gate {GATE_MS} ms "
              "(apply the confirmed budget arithmetic from Stage 1)")  # fmt: skip
    return case


def main(argv: list[str] | None = None) -> int:
    _summary = (__doc__ or "").strip().splitlines()
    ap = argparse.ArgumentParser(description=_summary[0] if _summary else None)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--collar-file", default=COLLAR_PATH)
    ap.add_argument("--label-error-ms", type=float, default=None)
    ap.add_argument("--allow-provisional", action="store_true")
    ap.add_argument("--out", default="bench-results/vad-decision")
    args = ap.parse_args(argv)

    p = Path(args.collar_file)
    if not p.exists():
        if not args.allow_provisional:
            print(
                f"no Stage 3B collar at {args.collar_file}; run `make vad-collar` first",
                file=sys.stderr,
            )
            return 1  # fmt: skip
        collar_s, partial, data = 0.150, True, {}
    else:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            collar_s = float(data["collar_s"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"cannot read {args.collar_file}: {exc}", file=sys.stderr)
            return 1
        partial = bool(data.get("partial", False))
    if partial and not args.allow_provisional:
        print(f"{args.collar_file} is partial; not a Stage 3B collar", file=sys.stderr)
        return 1
    label_error_s = (
        args.label_error_ms / 1000.0 if args.label_error_ms is not None else collar_s
    )

    thresholds, timeouts = list(THRESHOLD_GRID), list(TIMEOUT_GRID_MS)
    items = VadManifest.from_json(args.manifest).items
    rows, summaries = evaluate_pairs(
        items, thresholds, timeouts, collar_s, label_error_s
    )
    case = print_report(rows, summaries, thresholds, timeouts, collar_s, label_error_s)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with open(out / "cells.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "case": case,
                "collar_s": collar_s,
                "label_error_s": label_error_s,
                "pairs": [
                    {"threshold": t, "timeout_ms": o, **s}
                    for (t, o), s in summaries.items()
                ],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out / 'cells.csv'} and {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
