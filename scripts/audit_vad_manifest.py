"""Audit the VAD manifest and the gold template against the Stage 3A specification.

Run: make vad-audit
Checks counts, coverage, noise/RIR mapping, file existence, source bounds and the
ground-truth construction (16000-sample lead, GT = bounds length, gap = pause_ms * 16).
Exit status is 1 when any problem is found.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import soundfile as sf

from bench.vad_manifest import (
    CONDITION_SNR_LEVELS,
    CONDITIONS,
    PAUSE_LENGTHS_MS,
    VadManifest,
)

LEAD = 16000
NOISE = {
    "quiet": ("steady", False),
    "street": ("impulsive", False),
    "indoors": ("steady", True),
    "near-field": ("steady", True),
    "far-field": ("impulsive", True),
}
PER_CELL_PER_LANG = 10


def audit_manifest(path: str, per_cell_per_lang: int = PER_CELL_PER_LANG) -> dict:
    manifest = VadManifest.from_json(path)
    problems: list[str] = []
    cells: collections.Counter = collections.Counter()
    info: dict[str, tuple[int, int]] = {}

    def source(p: str) -> tuple[int, int]:
        if p not in info:
            i = sf.info(p)
            info[p] = (i.frames, i.samplerate)
        return info[p]

    for it in manifest.items:
        cells[(it.condition, it.snr_db, it.pause_ms, it.language)] += 1
        noise_type, uses_rir = NOISE[it.condition]
        if it.snr_db is None:
            if it.noise_type != "clean" or it.noise_path or it.rir_path:
                problems.append(f"{it.id}: clean item must have no noise and no RIR")
        else:
            if it.noise_type != noise_type or f"/musan/{noise_type}/" not in str(
                it.noise_path
            ):
                problems.append(f"{it.id}: wrong noise ({it.noise_type})")
            elif not Path(it.noise_path).exists():
                problems.append(f"{it.id}: noise file missing")
            if uses_rir != bool(it.rir_path):
                problems.append(f"{it.id}: RIR presence wrong")
            elif it.rir_path and (
                "/rirs/" not in it.rir_path or not Path(it.rir_path).exists()
            ):
                problems.append(f"{it.id}: RIR path wrong or missing")

        bounds = {"a": it.speech_a_bounds, "b": it.speech_b_bounds}
        if bounds["a"] is None or bounds["b"] is None:
            problems.append(f"{it.id}: missing source bounds")
            continue
        lengths = {}
        for key, src in (("a", it.clean_audio_path), ("b", it.speech_b_path)):
            if not src or not Path(src).exists():
                problems.append(f"{it.id}: missing source {src}")
                continue
            frames, rate = source(src)
            onset, offset = bounds[key]
            if rate != 16000:
                problems.append(f"{it.id}: source sample rate {rate}")
            if not 0 <= onset < offset <= frames:
                problems.append(f"{it.id}: bounds {bounds[key]} outside {frames}")
            lengths[key] = offset - onset
        if len(lengths) < 2 or len(it.ground_truth_segments) != 2:
            problems.append(f"{it.id}: cannot verify GT")
            continue
        (a0, a1), (b0, b1) = [tuple(s) for s in it.ground_truth_segments]
        if (
            a0 != LEAD
            or a1 - a0 != lengths["a"]
            or b1 - b0 != lengths["b"]
            or b0 - a1 != it.pause_ms * 16
        ):
            problems.append(f"{it.id}: GT does not match construction")

    expected = {
        (c, s, p, lang)
        for c in CONDITIONS
        for s in CONDITION_SNR_LEVELS[c]
        for p in PAUSE_LENGTHS_MS
        for lang in ("vi", "en")
    }
    off_cells = [k for k, v in cells.items() if v != per_cell_per_lang]
    return {
        "version": manifest.version,
        "total": len(manifest.items),
        "expected_total": 80 * per_cell_per_lang,
        "unique_ids": len({i.id for i in manifest.items}),
        "by_language": dict(collections.Counter(i.language for i in manifest.items)),
        "cells": len({k[:3] for k in cells}),
        "cells_off_size": len(off_cells),
        "cells_missing": len(expected - set(cells)),
        "cells_extra": len(set(cells) - expected),
        "unique_sources": len(info),
        "problems": problems,
    }


def audit_gold(path: str) -> dict:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    langs = collections.Counter(
        "vi" if "/vi/" in r["audio_path"] else "en" for r in rows
    )
    per_cell = collections.Counter((r["condition"], r["snr_db"]) for r in rows)
    problems = []
    if len(rows) != 50:
        problems.append(f"{len(rows)} rows, expected 50")
    if langs.get("vi") != 25 or langs.get("en") != 25:
        problems.append(f"language split {dict(langs)}, expected 25/25")
    if len(per_cell) != 10 or min(per_cell.values(), default=0) < 2:
        problems.append("gold cells: need 10 cells with >= 2 clips each")
    missing = sum(not Path(r["audio_path"]).exists() for r in rows)
    if missing:
        problems.append(f"{missing} gold audio files missing")
    return {
        "rows": len(rows),
        "by_language": dict(langs),
        "cells": len(per_cell),
        "min_per_cell": min(per_cell.values(), default=0),
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="eval_data/vad_manifest_v2.json")
    ap.add_argument("--gold", default="eval_data/vad_gold_template.csv")
    ap.add_argument("--per-cell-per-lang", type=int, default=PER_CELL_PER_LANG)
    args = ap.parse_args(argv)

    m = audit_manifest(args.manifest, args.per_cell_per_lang)
    g = audit_gold(args.gold)
    print(f"MANIFEST {m['version']}: total={m['total']} unique_ids={m['unique_ids']}")
    print(f"  by_language={m['by_language']} cells={m['cells']}")
    print(
        f"  cells_off_size={m['cells_off_size']} cells_missing={m['cells_missing']} "
        f"cells_extra={m['cells_extra']} unique_sources={m['unique_sources']}"
    )
    print(f"  problems={len(m['problems'])}")
    for line in m["problems"][:10]:
        print(f"    {line}")
    print(
        f"GOLD: rows={g['rows']} by_language={g['by_language']} cells={g['cells']} "
        f"min_per_cell={g['min_per_cell']} problems={len(g['problems'])}"
    )
    for line in g["problems"]:
        print(f"    {line}")
    bad = m["problems"] or g["problems"] or m["cells_off_size"] or m["cells_missing"]
    wrong_size = m["total"] != m["expected_total"] or m["unique_ids"] != m["total"]
    return 1 if bad or wrong_size else 0


if __name__ == "__main__":
    sys.exit(main())
