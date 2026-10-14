#!/usr/bin/env python3
"""Turn ``run_results.json`` files into the per-item CSVs under
``docs/benchmark/raw-results/``.

Those CSVs used to be produced by hand, which is why the report could quote a
number that no committed artifact supported (issue #110 review): nothing in the
repository recorded how a committed result had been produced. This script is that
missing step.

It also records the effective beam width, which the run records themselves do not
carry. ``bench/run.py`` does not write ``item.config`` back into the result, so a
committed CSV cannot say which beam produced a WER or a BLEU. Pass the beam
explicitly (``--run <path>=<beam>``); a path containing ``beam-N`` is picked up
automatically, which is what the exploratory sweep directories look like.

Examples
--------
Reproduce the MT table from the two full-pool runs plus the gold-set runs::

    uv run python -m bench.scripts.results_to_csv \\
        --out docs/benchmark/raw-results/fleurs-mt.csv --stage MT \\
        --run bench-results/fleurs-mt/full/opus-mt-vi-en-ct2-cpu=5 \\
        --run bench-results/fleurs-mt/full/m2m100-vi-en-ct2-cpu=4 \\
        --run bench-results/fleurs-mt/gold-set/opus-mt-vi-en-ct2-cpu=5 \\
        --run bench-results/fleurs-mt/gold-set/m2m100-vi-en-ct2-cpu=4

Sweep runs carry their beam in the path, so it needs no annotation::

    uv run python -m bench.scripts.results_to_csv \\
        --out /tmp/sweep.csv --run bench-results/archive/beam-sweep/fleurs/opus-mt-vi-en-ct2-cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

FIELDS = [
    "run",
    "item_id",
    "stage",
    "language",
    "direction",
    "snr",
    "noise_type",
    "candidate_id",
    "latency_s",
    "peak_ram_mb",
    "wer",
    "cer",
    "bleu",
    "rtf",
    "beam_size",
    "notes",
    "audio_ref",
    "input_text",
    "output_audio_path",
    "error",
]

_BEAM_DIR = re.compile(r"beam-(\d+)")


def render(value: object) -> str:
    """Render a JSON value the way the committed CSVs hold it.

    Floats keep full repr precision (the existing files store 0.21739130434782608,
    not a rounded 0.2174); None becomes an empty cell; a notes list is joined so a
    single cell never breaks the row.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value)


def run_label(run_dir: Path, results_root: Path) -> str:
    """Label a run the way the committed CSVs do: path under the results root.

    ``bench-results/fleurs-mt/full/opus-mt-vi-en-ct2-cpu`` becomes
    ``fleurs-mt/full/opus-mt-vi-en-ct2-cpu``.
    """
    rel = run_dir.resolve().relative_to(results_root.resolve())
    return rel.as_posix()


def beam_for(run_dir: Path, explicit: str | None) -> str:
    """Beam width from the ``=N`` suffix, else from a ``beam-N`` path segment."""
    if explicit is not None:
        return explicit
    for part in run_dir.parts:
        m = _BEAM_DIR.fullmatch(part)
        if m:
            return m.group(1)
    return ""


def rows_for(
    run_dir: Path,
    results_root: Path,
    beam: str | None,
    drop_errors: bool,
    stage: str | None,
) -> list[dict]:
    path = run_dir / "run_results.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    label = run_label(run_dir, results_root)
    width = beam_for(run_dir, beam)
    out: list[dict] = []
    for rec in records:
        item, result, metrics = rec["item"], rec["result"], rec["metrics"]
        if drop_errors and result.get("error"):
            continue
        # A run directory can hold more than the stage it contributes to: the
        # MT "gold-set" runs were pointed at eval_manifest_v1.json and returned
        # all 594 items (540 ASR + 42 MT + 12 TTS). The committed CSV keeps the
        # 42 MT rows, so the stage filter has to be explicit rather than implied
        # by the directory name.
        if stage is not None and item.get("stage") != stage:
            continue
        out.append(
            {
                "run": label,
                "item_id": render(item.get("id")),
                "stage": render(item.get("stage")),
                "language": render(item.get("language")),
                "direction": render(item.get("direction")),
                "snr": render(item.get("snr")),
                "noise_type": render(item.get("noise_type")),
                "candidate_id": render(result.get("candidate_id")),
                "latency_s": render(result.get("latency_s")),
                "peak_ram_mb": render(result.get("peak_ram_mb")),
                "wer": render(metrics.get("wer")),
                "cer": render(metrics.get("cer")),
                "bleu": render(metrics.get("bleu")),
                "rtf": render(metrics.get("rtf")),
                "beam_size": width,
                "notes": render(metrics.get("notes")),
                "audio_ref": render(item.get("audio_ref")),
                "input_text": render(item.get("input_text")),
                "output_audio_path": render(result.get("output_audio_path")),
                "error": render(result.get("error")),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="run_results.json -> per-item CSV")
    ap.add_argument("--out", required=True, help="CSV path to write")
    ap.add_argument(
        "--results-root",
        default="bench-results",
        help="directory the run labels are relative to (default: bench-results)",
    )
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="PATH[=BEAM]",
        help="a run directory, repeat per candidate; =BEAM records the beam width",
    )
    ap.add_argument(
        "--keep-errors",
        action="store_true",
        help="keep rows whose result carries an error (default: drop them)",
    )
    ap.add_argument(
        "--stage",
        help="keep only items of this stage (ASR/MT/TTS); run directories are "
        "not stage-scoped, so this is usually required",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    rows: list[dict] = []
    for spec in args.run:
        beam: str | None = None
        if "=" in spec:
            spec, beam = spec.rsplit("=", 1)
        run_dir = Path(spec)
        if not (run_dir / "run_results.json").exists():
            ap.error(f"no run_results.json under {run_dir}")
        rows.extend(
            rows_for(run_dir, results_root, beam, not args.keep_errors, args.stage)
        )

    if not rows:
        ap.error("no rows produced")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: the committed CSVs carry a BOM, and aggregate_results.py reads
    # them with utf-8-sig. Keep the two in step.
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    runs = sorted({r["run"] for r in rows})
    print(f"wrote {out} — {len(rows)} rows across {len(runs)} runs")
    for r in runs:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
