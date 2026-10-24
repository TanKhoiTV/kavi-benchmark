"""Stage 5: run the full sweep and write the compact results table and command record.

Requires a Stage 3B collar (bench/vad_collar.json, not partial). Raw outputs go to
bench-results/ (gitignored); only the compact table and the command record are meant
to be committed. The run is verified: row count against grid x conditions x SNR x
pauses, minimum items per cell, empty cells, and the manifest seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.vad_manifest import (
    CONDITION_SNR_LEVELS,
    CONDITIONS,
    PAUSE_LENGTHS_MS,
    VadManifest,
    build_vad_manifest,
)
from bench.vad_runner import (
    COLLAR_PATH,
    DEFAULT_MANIFEST,
    THRESHOLD_GRID,
    TIMEOUT_GRID_MS,
    ItemResult,
)
from bench.vad_sweep import run_sweep

TABLE_PATH = "bench/results/vad/sweep_table.csv"
RECORD_PATH = "bench/results/vad/sweep_record.md"
RAW_DIR = "bench-results/vad-stage5"
MIN_ITEMS_PER_CELL = 20
MANIFEST_SEED = 42

FIELDS = [
    "condition", "snr_db", "pause_ms", "threshold", "timeout_ms", "items",
    "false_trigger_rate_per_min", "missed_onset_rate",
    "clipped_start_ms_mean", "clipped_end_ms_mean", "split_utterance_rate",
    "junction_split_rate", "eou_delay_median_ms", "eou_delay_p95_ms", "eou_delay_max_ms",
]  # fmt: skip


def expected_cells() -> list[tuple[str, float | None, int]]:
    return [
        (c, s, p)
        for c in CONDITIONS
        for s in CONDITION_SNR_LEVELS[c]
        for p in PAUSE_LENGTHS_MS
    ]


def cell_row(thr: float, to: int, key: tuple, rs: list[ItemResult]) -> dict:
    condition, snr, pause = key
    non_speech_min = sum(r.non_speech_s for r in rs) / 60.0
    ft = sum(r.metrics.false_trigger_count for r in rs)
    eou = [r.metrics.end_of_utterance_delay_ms for r in rs]
    n = len(rs)
    return {
        "condition": condition,
        "snr_db": "clean" if snr is None else snr,
        "pause_ms": pause,
        "threshold": thr,
        "timeout_ms": to,
        "items": n,
        "false_trigger_rate_per_min": ft / non_speech_min if non_speech_min > 0 else "",
        "missed_onset_rate": sum(r.metrics.missed_onset_count for r in rs) / n,
        "clipped_start_ms_mean": float(
            np.mean([r.metrics.clipped_start_ms for r in rs])
        ),
        "clipped_end_ms_mean": float(np.mean([r.metrics.clipped_end_ms for r in rs])),
        "split_utterance_rate": sum(r.metrics.split_count > 0 for r in rs) / n,
        "junction_split_rate": sum(r.junction_split for r in rs) / n,
        "eou_delay_median_ms": float(np.median(eou)),
        "eou_delay_p95_ms": float(np.percentile(eou, 95)),
        "eou_delay_max_ms": float(max(eou)),
    }


def build_rows(results: dict[tuple[float, int], list[ItemResult]]) -> list[dict]:
    order = {cell: i for i, cell in enumerate(expected_cells())}
    rows = []
    for (thr, to), rs in results.items():
        groups: dict[tuple, list[ItemResult]] = defaultdict(list)
        for r in rs:
            groups[(r.condition, r.snr_db, r.pause_ms)].append(r)
        for key in sorted(groups, key=lambda k: order.get(k, len(order))):
            rows.append(cell_row(thr, to, key, groups[key]))
    return rows


def verify_run(
    rows: list[dict],
    thresholds: list[float],
    timeouts: list[int],
    cells: list[tuple],
    min_items: int,
) -> dict:
    expected = len(thresholds) * len(timeouts) * len(cells)
    seen = {
        (r["threshold"], r["timeout_ms"], r["condition"], r["snr_db"], r["pause_ms"])
        for r in rows
    }
    missing = [
        (t, o, c, "clean" if s is None else s, p)
        for t in thresholds
        for o in timeouts
        for c, s, p in cells
        if (t, o, c, "clean" if s is None else s, p) not in seen
    ]
    below = [r for r in rows if r["items"] < min_items]
    return {
        "expected_rows": expected,
        "actual_rows": len(rows),
        "duplicate_rows": len(rows) - len(seen),
        "missing_cells": missing,
        "cells_below_minimum": [
            (r["threshold"], r["timeout_ms"], r["condition"], r["snr_db"], r["pause_ms"], r["items"])
            for r in below
        ],
        "ok": len(rows) == expected and not missing and not below and len(seen) == len(rows),
    }  # fmt: skip


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return out.stdout.strip()


def read_collar(path: str) -> tuple[float, bool, dict]:
    p = Path(path)
    if not p.exists():
        return 0.150, True, {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return float(data["collar_s"]), bool(data.get("partial", False)), data


def seed_check(manifest_path: str, seed: int) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "m.json"
        build_vad_manifest(out_path=str(out), seed=seed)
        return out.read_bytes() == Path(manifest_path).read_bytes()


def write_table(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_record(path: str, info: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    v = info["verification"]
    lines = [
        "# VAD sweep record (Stage 5)",
        "",
        f"- command: `{info['command']}`",
        f"- git commit: `{info['commit']}`"
        + (" (working tree dirty)" if info["dirty"] else ""),
        f"- manifest: `{info['manifest']}` sha256 `{info['manifest_sha256']}`",
        f"- manifest seed: {info['seed']} (regenerated from the seed: "
        f"{info['seed_reproduced']})",
        f"- source bounds: `bench/vad_source_bounds.json` sha256 `{info['bounds_sha256']}`",
        f"- collar: {info['collar_s']} s from `{info['collar_file']}` sha256 `{info['collar_sha256']}`"
        + (" PROVISIONAL" if info["provisional"] else ""),
        f"- grid: thresholds {info['thresholds']}, timeouts {info['timeouts']} ms",
        f"- minimum items per cell: {info['min_items']}",
        f"- python {info['python']}, numpy {info['numpy']}",
        f"- runtime: {info['runtime_s']:.1f} s",
        "",
        "## Verification",
        "",
        f"- rows: {v['actual_rows']} (expected {v['expected_rows']} = "
        f"{len(info['thresholds'])} thresholds x {len(info['timeouts'])} timeouts x "
        f"{info['n_cells']} condition/SNR/pause cells)",
        f"- missing cells: {len(v['missing_cells'])}",
        f"- cells below the minimum item count: {len(v['cells_below_minimum'])}",
        f"- failed runs: {info['failed_runs']}",
        f"- result: {'PASS' if v['ok'] else 'FAIL'}",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--collar-file", default=COLLAR_PATH)
    ap.add_argument("--table", default=TABLE_PATH)
    ap.add_argument("--record", default=RECORD_PATH)
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--min-items", type=int, default=MIN_ITEMS_PER_CELL)
    ap.add_argument("--seed", type=int, default=MANIFEST_SEED)
    ap.add_argument("--skip-seed-check", action="store_true")
    ap.add_argument("--allow-provisional", action="store_true")
    ap.add_argument("--publish-provisional", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    collar_s, provisional, collar_data = read_collar(args.collar_file)
    if provisional and not args.allow_provisional:
        print(
            f"no Stage 3B collar at {args.collar_file} (or it is partial); "
            "finish the 50 labels and run `make vad-collar` first",
            file=sys.stderr,
        )
        return 1
    if collar_data.get("partial") and not args.allow_provisional:
        print(
            f"{args.collar_file} is marked partial; not a Stage 3B collar",
            file=sys.stderr,
        )
        return 1

    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    check_dirty = not args.allow_provisional or args.publish_provisional
    if dirty and not args.allow_dirty and check_dirty:
        print(
            "tracked files have uncommitted changes; commit them so the record names the code used",
            file=sys.stderr,
        )
        return 1  # fmt: skip

    thresholds, timeouts = list(THRESHOLD_GRID), list(TIMEOUT_GRID_MS)
    manifest = VadManifest.from_json(args.manifest)
    started = time.perf_counter()
    failed_runs = 0
    try:
        results, _ = run_sweep(manifest.items, thresholds, timeouts, collar_s)
    except Exception as exc:  # noqa: BLE001
        print(f"sweep failed: {exc}", file=sys.stderr)
        return 1
    rows = build_rows(results)
    cells = expected_cells()
    verification = verify_run(rows, thresholds, timeouts, cells, args.min_items)
    reproduced = None if args.skip_seed_check else seed_check(args.manifest, args.seed)

    info = {
        "command": "python -m bench.vad_stage5 " + " ".join(argv if argv is not None else sys.argv[1:]),
        "commit": git("rev-parse", "HEAD"),
        "dirty": dirty,
        "manifest": args.manifest,
        "manifest_sha256": sha256(args.manifest),
        "seed": args.seed,
        "seed_reproduced": "skipped" if reproduced is None else ("yes" if reproduced else "NO"),
        "bounds_sha256": sha256("bench/vad_source_bounds.json"),
        "collar_s": collar_s,
        "collar_file": args.collar_file,
        "collar_sha256": sha256(args.collar_file) if Path(args.collar_file).exists() else "n/a",
        "provisional": provisional,
        "thresholds": thresholds,
        "timeouts": timeouts,
        "min_items": args.min_items,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "runtime_s": time.perf_counter() - started,
        "failed_runs": failed_runs,
        "n_cells": len(cells),
        "verification": verification,
    }  # fmt: skip

    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    table, record = (args.table, args.record)
    if provisional and not args.publish_provisional:
        table, record = str(raw / "sweep_table.csv"), str(raw / "sweep_record.md")

    print(
        f"rows: {verification['actual_rows']} / expected {verification['expected_rows']}"
    )
    print(f"missing cells: {len(verification['missing_cells'])}")
    print(
        f"cells below {args.min_items} items: {len(verification['cells_below_minimum'])}"
    )
    print(f"failed runs: {failed_runs}")
    print(f"manifest seed {args.seed} reproduced: {info['seed_reproduced']}")
    if provisional and not args.publish_provisional:
        print("collar is PROVISIONAL: outputs written to bench-results/ only")
    elif provisional:
        print("collar is PROVISIONAL: published outputs are labelled PROVISIONAL")
    if reproduced is False:
        verification["ok"] = False
    if not verification["ok"]:
        for cell in verification["missing_cells"][:10]:
            print("  missing", cell)
        for cell in verification["cells_below_minimum"][:10]:
            print("  below minimum", cell)
        write_record(record, info)
        print(
            f"VERIFICATION FAILED; record written to {record}, table not written",
            file=sys.stderr,
        )
        return 1

    write_table(table, rows)
    write_record(record, info)
    print(f"wrote {table} and {record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
