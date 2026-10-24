"""VAD-only runner: Stage 3A manifest -> timeline audio -> energy VAD -> Stage 2 scorer.

One (energy_threshold, speech_timeout_ms) configuration per invocation. No ASR, MT
or TTS is involved. Audio is built in memory with build_timeline_audio and is not
written to disk.

Ground truth handed to the scorer is the whole A + pause + B span as ONE
utterance: the manifest keeps A and B as two segments for construction audits, but
the scorer's split metric is defined on a gap inside one ground-truth utterance.
Non-speech time is therefore the leading plus trailing silence only.

The collar is read from bench/vad_collar.json (key "collar_s"). Until Stage 3B
writes that file, the Stage 2 scorer default of 0.150 s is used and every output
is labelled PROVISIONAL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench.vad_energy import (
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_SPEECH_TIMEOUT_MS,
    EnergyVad,
    EnergyVadResult,
)
from bench.vad_manifest import VadItem, VadManifest
from bench.vad_noise import SR, TARGET_P99_PEAK, build_timeline_audio, load_mono
from bench.vad_scorer import VadMetrics, score_vad_item

DEFAULT_MANIFEST = "eval_data/vad_manifest_v2.json"
COLLAR_PATH = "bench/vad_collar.json"
PROVISIONAL_COLLAR_S = 0.150
THRESHOLD_GRID = [0.01, 0.02, 0.05, 0.10, 0.20]
TIMEOUT_GRID_MS = [200, 350, 500, 750, 1000]


@dataclass
class ItemResult:
    id: str
    condition: str
    snr_db: float | None
    pause_ms: int
    language: str
    metrics: VadMetrics
    non_speech_s: float
    n_vad_segments: int
    eos_delay_ms: float | None
    violation: bool
    junction_split: bool
    junction_violation: bool


def load_collar(path: str = COLLAR_PATH) -> tuple[float, bool]:
    p = Path(path)
    if not p.exists():
        return PROVISIONAL_COLLAR_S, True
    value = json.loads(p.read_text(encoding="utf-8")).get("collar_s")
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{path}: 'collar_s' must be a positive number")
    return float(value), False


def select_items(items: list[VadItem], per_cell_per_lang: int | None) -> list[VadItem]:
    if per_cell_per_lang is None:
        return list(items)
    groups: dict[tuple, list[VadItem]] = defaultdict(list)
    for it in items:
        groups[(it.condition, it.snr_db, it.pause_ms, it.language)].append(it)
    keep: set[str] = set()
    for group in groups.values():
        for it in sorted(group, key=lambda x: x.id)[:per_cell_per_lang]:
            keep.add(it.id)
    return [it for it in items if it.id in keep]


def merged_gt_s(item: VadItem) -> list[tuple[float, float]]:
    segs = item.ground_truth_segments
    if not segs:
        raise ValueError(f"{item.id}: no ground-truth segments")
    return [(segs[0][0] / SR, segs[-1][1] / SR)]


def _bounds(value) -> tuple[int, int] | None:
    return None if value is None else (int(value[0]), int(value[1]))


class _AudioCache:
    def __init__(self) -> None:
        self._data: dict[str, np.ndarray] = {}

    def get(self, path: str) -> np.ndarray:
        if path not in self._data:
            self._data[path] = load_mono(path)
        return self._data[path]


@dataclass
class PreparedItem:
    item: VadItem
    energy: np.ndarray
    n_samples: int
    gt: list[tuple[float, float]]
    total_s: float
    non_speech_s: float
    a_end_s: float
    b_start_s: float


def prepare_item(
    item: VadItem, cache: _AudioCache, timing: dict[str, float]
) -> PreparedItem:
    """Build the item's audio once and keep only what the VAD needs: frame energy."""
    if item.speech_b_path is None:
        raise ValueError(f"{item.id}: speech_b_path missing")

    t0 = time.perf_counter()
    built = build_timeline_audio(
        cache.get(item.clean_audio_path),
        cache.get(item.speech_b_path),
        item.pause_ms,
        noise=cache.get(item.noise_path) if item.noise_path else None,
        rir=cache.get(item.rir_path) if item.rir_path else None,
        snr_db=item.snr_db,
        bounds_a=_bounds(item.speech_a_bounds),
        bounds_b=_bounds(item.speech_b_bounds),
    )
    if [tuple(s) for s in item.ground_truth_segments] != built.gt_segments:
        raise RuntimeError(
            f"{item.id}: manifest GT differs from the built timeline "
            "(stale manifest or source-bounds file?)"
        )
    energy = EnergyVad().frame_energy(built.audio)
    timing["build_s"] += time.perf_counter() - t0

    gt = merged_gt_s(item)
    total_s = len(built.audio) / SR
    (_, a_end_sample), (b_start_sample, _) = item.ground_truth_segments[:2]
    return PreparedItem(
        item=item,
        energy=energy,
        n_samples=len(built.audio),
        gt=gt,
        total_s=total_s,
        non_speech_s=total_s - (gt[0][1] - gt[0][0]),
        a_end_s=a_end_sample / SR,
        b_start_s=b_start_sample / SR,
    )


def score_prepared(
    prep: PreparedItem,
    vad: EnergyVad,
    collar_s: float,
    timing: dict[str, float],
    result: EnergyVadResult | None = None,
) -> ItemResult:
    item = prep.item
    t1 = time.perf_counter()
    if result is None:
        result = vad.detect_energy(prep.energy, prep.n_samples)
    t2 = time.perf_counter()
    metrics = score_vad_item(
        gt_segments=prep.gt,
        vad_segments=result.segments,
        total_duration_s=prep.total_s,
        non_speech_duration_s=prep.non_speech_s,
        collar=collar_s,
        speech_timeout_s=vad.speech_timeout_ms / 1000.0,
    )
    t3 = time.perf_counter()
    if metrics.errors:
        raise RuntimeError(f"{item.id}: scorer errors {metrics.errors}")
    timing["vad_s"] += t2 - t1
    timing["score_s"] += t3 - t2

    gt_start, gt_end = prep.gt[0]
    junction_split = not any(
        seg[0] <= prep.a_end_s and seg[1] >= prep.b_start_s for seg in result.segments
    )
    overlapping = [s for s in result.segments if s[0] < gt_end and s[1] > gt_start]
    eos_delay_ms = None
    if result.eos_s and result.eos_s[-1] is not None:
        eos_delay_ms = (result.eos_s[-1] - gt_end) * 1000.0

    return ItemResult(
        id=item.id,
        condition=item.condition,
        snr_db=item.snr_db,
        pause_ms=item.pause_ms,
        language=item.language,
        metrics=metrics,
        non_speech_s=prep.non_speech_s,
        n_vad_segments=len(result.segments),
        eos_delay_ms=eos_delay_ms,
        violation=item.pause_ms <= vad.speech_timeout_ms and len(overlapping) > 1,
        junction_split=junction_split,
        junction_violation=junction_split and item.pause_ms <= vad.speech_timeout_ms,
    )


def evaluate_item(
    item: VadItem,
    vad: EnergyVad,
    collar_s: float,
    cache: _AudioCache,
    timing: dict[str, float],
) -> ItemResult:
    return score_prepared(prepare_item(item, cache, timing), vad, collar_s, timing)


def run_config(
    items: list[VadItem],
    threshold: float,
    timeout_ms: int,
    collar_s: float,
) -> tuple[list[ItemResult], dict[str, float]]:
    vad = EnergyVad(threshold, timeout_ms)
    cache = _AudioCache()
    timing = {"build_s": 0.0, "vad_s": 0.0, "score_s": 0.0}
    t0 = time.perf_counter()
    results = [evaluate_item(it, vad, collar_s, cache, timing) for it in items]
    timing["wall_s"] = time.perf_counter() - t0
    return results, timing


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _stats(values: list[float]) -> dict[str, float | None]:
    return {
        "median": _pct(values, 50),
        "p95": _pct(values, 95),
        "max": float(max(values)) if values else None,
    }


def summarize(results: list[ItemResult]) -> dict:
    n = len(results)
    non_speech_min = sum(r.non_speech_s for r in results) / 60.0
    false_triggers = sum(r.metrics.false_trigger_count for r in results)
    clipped = [r.metrics.clipped_start_ms + r.metrics.clipped_end_ms for r in results]
    by_pause: dict[int, list[bool]] = defaultdict(list)
    for r in results:
        by_pause[r.pause_ms].append(r.metrics.split_count > 0)
    eos = [r.eos_delay_ms for r in results if r.eos_delay_ms is not None]
    by_pause_junction: dict[int, list[bool]] = defaultdict(list)
    for r in results:
        by_pause_junction[r.pause_ms].append(r.junction_split)
    return {
        "n_items": n,
        "false_trigger_count": false_triggers,
        "false_trigger_rate_per_min": (
            false_triggers / non_speech_min if non_speech_min > 0 else None
        ),
        "missed_onset_rate": sum(r.metrics.missed_onset_count for r in results) / n,
        "clipped_ms_per_utterance": {
            "mean": float(np.mean(clipped)),
            "p95": _pct(clipped, 95),
            "max": float(max(clipped)),
        },
        "split_rate_by_pause": {
            str(p): sum(v) / len(v) for p, v in sorted(by_pause.items())
        },
        "pause_violations": sum(r.violation for r in results),
        "utterance_split_rate": sum(r.metrics.split_count > 0 for r in results) / n,
        "junction_split_rate_by_pause": {
            str(p): sum(v) / len(v) for p, v in sorted(by_pause_junction.items())
        },
        "junction_violations": sum(r.junction_violation for r in results),
        "eou_delay_ms": _stats([r.metrics.end_of_utterance_delay_ms for r in results]),
        "eos_delay_ms": _stats(eos),
        "eos_undeclared": n - len(eos),
    }


def aggregate(results: list[ItemResult]) -> dict:
    cells: dict[tuple, list[ItemResult]] = {}
    for r in results:
        cells.setdefault((r.condition, r.snr_db), []).append(r)
    return {
        "cells": [
            {"condition": c, "snr_db": s, **summarize(rs)}
            for (c, s), rs in cells.items()
        ],
        "overall": summarize(results),
    }


def estimate_sweep(
    timing: dict[str, float], n_run: int, n_full: int, n_configs: int
) -> dict[str, float]:
    build = timing["build_s"] / n_run
    rest = (timing["vad_s"] + timing["score_s"]) / n_run
    return {
        "per_item_build_ms": build * 1000,
        "per_item_vad_score_ms": rest * 1000,
        "n_configs": n_configs,
        "n_items_full": n_full,
        "rebuild_audio_every_config_s": n_configs * n_full * (build + rest),
        "build_audio_once_s": n_full * build + n_configs * n_full * rest,
    }


def _dur(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 120 else f"{seconds / 60:.1f}min"


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def print_report(
    summary: dict,
    config: dict,
    collar_s: float,
    provisional: bool,
    timing: dict,
    estimate: dict,
) -> None:
    tag = " (PROVISIONAL, Stage 3B not run)" if provisional else ""
    print(
        f"config: energy_threshold={config['threshold']} "
        f"speech_timeout_ms={config['timeout_ms']} | items={config['n_items']}"
    )
    print(f"collar: {collar_s:.3f}s{tag}")
    print(f"speech level: each clip scaled to p99 frame peak {TARGET_P99_PEAK}")
    print(
        f"source bounds: {config['n_trimmed']}/{config['n_items']} items use "
        "reference-VAD speech bounds"
    )
    print("reference limits: false-trigger<=0.1/min, missed-onset<=5%, clip<=50ms/utt")
    print(
        "jsplit/jviol: VAD did not bridge the A-B pause (jviol = pause <= timeout); "
        "anySpl%: any gap > timeout anywhere in the utterance"
    )
    head = (
        f"{'cond/snr':<16}{'n':>4}{'FT/min':>8}{'miss%':>7}{'clipMean':>9}"
        f"{'clipP95':>8}{'clipMax':>8}{'jsplit% by pause':>20}{'jviol':>6}{'anySpl%':>8}"
        f"{'EOUmed':>8}{'EOUp95':>8}{'EOSmed':>8}{'noEOS':>6}"
    )
    print(head)
    print("-" * len(head))
    rows = [(f"{c['condition']}/{c['snr_db']}", c) for c in summary["cells"]]
    rows.append(("OVERALL", summary["overall"]))
    for label, c in rows:
        split = "/".join(
            f"{v * 100:.0f}" for v in c["junction_split_rate_by_pause"].values()
        )
        print(
            f"{label:<16}{c['n_items']:>4}"
            f"{_fmt(c['false_trigger_rate_per_min'], '.2f'):>8}"
            f"{c['missed_onset_rate'] * 100:>7.1f}"
            f"{c['clipped_ms_per_utterance']['mean']:>9.1f}"
            f"{_fmt(c['clipped_ms_per_utterance']['p95'], '.1f'):>8}"
            f"{c['clipped_ms_per_utterance']['max']:>8.1f}"
            f"{split:>20}{c['junction_violations']:>6}"
            f"{c['utterance_split_rate'] * 100:>8.0f}"
            f"{_fmt(c['eou_delay_ms']['median'], '.0f'):>8}"
            f"{_fmt(c['eou_delay_ms']['p95'], '.0f'):>8}"
            f"{_fmt(c['eos_delay_ms']['median'], '.0f'):>8}"
            f"{c['eos_undeclared']:>6}"
        )
    print()
    print(
        f"runtime: wall={_dur(timing['wall_s'])} build={_dur(timing['build_s'])} "
        f"vad={_dur(timing['vad_s'])} score={_dur(timing['score_s'])} "
        f"({timing['wall_s'] / config['n_items'] * 1000:.0f} ms/item)"
    )
    print(
        f"full sweep estimate ({estimate['n_configs']} configs x "
        f"{estimate['n_items_full']} items): "
        f"rebuild audio per config={_dur(estimate['rebuild_audio_every_config_s'])}, "
        f"build audio once={_dur(estimate['build_audio_once_s'])}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--threshold", type=float, default=DEFAULT_ENERGY_THRESHOLD)
    ap.add_argument("--timeout-ms", type=int, default=DEFAULT_SPEECH_TIMEOUT_MS)
    ap.add_argument("--per-cell-per-lang", type=int, default=None)
    ap.add_argument("--collar-file", default=COLLAR_PATH)
    ap.add_argument("--out", default="bench-results/vad")
    args = ap.parse_args(argv)

    manifest = VadManifest.from_json(args.manifest)
    items = select_items(manifest.items, args.per_cell_per_lang)
    collar_s, provisional = load_collar(args.collar_file)
    results, timing = run_config(items, args.threshold, args.timeout_ms, collar_s)
    summary = aggregate(results)
    n_configs = len(THRESHOLD_GRID) * len(TIMEOUT_GRID_MS)
    estimate = estimate_sweep(timing, len(items), len(manifest.items), n_configs)
    config = {
        "threshold": args.threshold,
        "timeout_ms": args.timeout_ms,
        "n_items": len(items),
        "per_cell_per_lang": args.per_cell_per_lang,
        "target_p99_peak": TARGET_P99_PEAK,
        "n_trimmed": sum(1 for it in items if it.speech_a_bounds is not None),
    }

    print_report(summary, config, collar_s, provisional, timing, estimate)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vad_thr{args.threshold:g}_to{args.timeout_ms}.json"
    out_path.write_text(
        json.dumps(
            {
                "config": config,
                "collar_s": collar_s,
                "collar_provisional": provisional,
                "timing": timing,
                "sweep_estimate": estimate,
                **summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
