"""Which statistic should decide the direction? Five candidates, one decode pass.

The Vietnamese-share sweep (`lid_share_sweep.py`) showed that
`argmax(mean per-token log-prob)` is biased toward English and needs ~70%
Vietnamese audio before it reliably says Vietnamese. The speaker- and
language-recognition literature says why: a threshold applied to
un-normalised scores from two heterogeneous systems is the textbook
failure, and the cure is per-model normalisation followed by calibration
(see `docs/reference/lid-code-switching-survey.md`).

This module tests that claim on the same audio, so the comparison is
apples-to-apples. It decodes each clip once with both shipping
checkpoints and records the raw per-token log-probs, then computes every
candidate statistic *post hoc* from those arrays. Recording the arrays
rather than a summary is deliberate: a new statistic can be evaluated
without re-decoding.

Candidates, all from the same token arrays:

    mean      argmax of the mean per-token log-prob. The current rule.
    median    argmax of the median token. One confident span cannot run
              away with the decision.
    topk      argmax of the mean of the most confident 30% of tokens.
    znorm     per-model z-score against that model's own clean-audio
              distribution, then subtract - the shape S-norm prescribes.
              Fitted on labelled clean audio only.
    tnorm     the margin over its own standard error,
              sqrt(var_vi/n_vi + var_en/n_en). This is the one that faces
              the recorded confound that English emits ~3-4x the tokens
              of Vietnamese for the same audio.

The z-score reference is fitted on the pure conditions of the same run,
so no mixed audio is used to tune what later scores mixed audio.

Every number here is decision quality, never latency: turnaround is a
device measurement on one clock, and nothing in this file records time.

Usage::

    uv run --with sherpa-onnx==2.0.1 --with sherpa-onnx-core==2.0.1 \\
        python -m bench.lid_statistic --pairs 40 --workers 6

Writes `bench-results/lid-statistic/report.json`: provenance, the fitted
reference, per-statistic results and the raw per-clip token arrays.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .lid_codeswitch import (
    EN_MODEL,
    EN_WAVS,
    THIN,
    VI_MODEL,
    VI_WAVS,
    load_recognizers,
    read_wav,
)
from .lid_share_sweep import CONSTRUCTIONS, build, wilson

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "bench-results/lid-statistic/report.json"

#: Enough shares to see the curve turn, few enough that a full run fits in
#: one sitting. 0 and 1 are pure conditions and carry a single pass.
SHARES = (0.0, 0.05, 0.15, 0.25, 0.40, 0.70, 1.0)
TOP_FRACTION = 0.30

_WORKER: dict = {}


def decode_tokens(recognizer, samples: np.ndarray, sr: int) -> list[float]:
    """Every token's log-probability, in decoding order.

    An empty list is a measurement, not a gap: on pure English the
    Vietnamese transducer emits nothing at all, which is the strongest
    evidence in the comparison.
    """
    stream = recognizer.create_stream()
    stream.accept_waveform(sr, samples)
    recognizer.decode_stream(stream)
    return [float(x) for x in stream.result.ys_log_probs]


def _init_worker() -> None:
    vi_rec, en_rec = load_recognizers()
    _WORKER["vi"] = vi_rec
    _WORKER["en"] = en_rec


def _run_condition(task: tuple[float, str, list[tuple[str, str]], int]) -> dict:
    share, construction, pairs, gap_ms = task
    rows = []
    for i, (vi_path, en_path) in enumerate(pairs):
        if share == 1.0:
            raw, sr = read_wav(Path(vi_path))
            mixed, seconds = raw, len(raw) / sr
        elif share == 0.0:
            raw, sr = read_wav(Path(en_path))
            mixed, seconds = raw, len(raw) / sr
        else:
            vi, sr = read_wav(Path(vi_path))
            en, _ = read_wav(Path(en_path))
            mixed = build(
                vi, en, sr, vi_share=share, construction=construction, gap_ms=gap_ms
            )
            seconds = len(mixed) / sr
        rows.append(
            {
                "key": f"{share:g}|{construction}|{i}",
                "share": share,
                "construction": construction,
                "seconds": round(seconds, 3),
                "vi": decode_tokens(_WORKER["vi"], mixed, sr),
                "en": decode_tokens(_WORKER["en"], mixed, sr),
            }
        )
    return {"share": share, "construction": construction, "rows": rows}


# --- the candidates -------------------------------------------------------
# Each takes a clip's two token lists and the fitted reference, and returns a
# margin where positive favours Vietnamese. Empties are not their business:
# `decide` owns that rule, in one place, because getting it wrong in one
# statistic and right in another is how a comparison becomes invalid.


def stat_mean(vi, en, ref) -> float:
    return statistics.fmean(vi) - statistics.fmean(en)


def stat_median(vi, en, ref) -> float:
    return statistics.median(vi) - statistics.median(en)


def stat_topk(vi, en, ref) -> float:
    def top(xs: list[float]) -> float:
        k = max(1, round(TOP_FRACTION * len(xs)))
        return statistics.fmean(sorted(xs, reverse=True)[:k])

    return top(vi) - top(en)


def stat_znorm(vi, en, ref) -> float:
    z_vi = (statistics.fmean(vi) - ref["vi_mean"]) / ref["vi_sd"]
    z_en = (statistics.fmean(en) - ref["en_mean"]) / ref["en_sd"]
    return z_vi - z_en


def stat_tnorm(vi, en, ref) -> float:
    if len(vi) < 2 or len(en) < 2:
        return stat_mean(vi, en, ref)
    d = statistics.fmean(vi) - statistics.fmean(en)
    se = math.sqrt(
        statistics.variance(vi) / len(vi) + statistics.variance(en) / len(en)
    )
    return d / se if se > 0 else d


def decide(fn, vi, en, ref) -> float:
    """One margin, with the empty-hypothesis rule applied in a single place.

    An empty hypothesis is the strongest evidence in the comparison, not a
    missing measurement: the English transducer emits nothing on ~9% of
    clean Vietnamese clips, and the Vietnamese one on ~1% of clean English
    clips. Treating that as "this statistic failed" inverts the signal - a
    version of this module did exactly that and reported 90% clean-VI
    recall where the truth is 99%, which is why the rule lives here.
    """
    if not vi and not en:
        return 0.0  # no evidence either way: the ladder should defer
    if not vi:
        return -math.inf
    if not en:
        return math.inf
    return fn(vi, en, ref)


CANDIDATES = {
    "mean": stat_mean,
    "median": stat_median,
    "topk": stat_topk,
    "znorm": stat_znorm,
    "tnorm": stat_tnorm,
}


def fit_reference(rows: list[dict]) -> dict:
    """Fit the z-score reference on the pure conditions only.

    The Vietnamese clips give the Vietnamese model's own-language
    distribution, the English clips the English model's. No mixed audio
    contributes, so the reference cannot be tuned to what it scores.
    """
    vi_clean = [
        statistics.fmean(r["vi"]) for r in rows if r["share"] == 1.0 and r["vi"]
    ]
    en_clean = [
        statistics.fmean(r["en"]) for r in rows if r["share"] == 0.0 and r["en"]
    ]
    return {
        "vi_mean": statistics.fmean(vi_clean),
        "vi_sd": statistics.pstdev(vi_clean) or 1e-9,
        "en_mean": statistics.fmean(en_clean),
        "en_sd": statistics.pstdev(en_clean) or 1e-9,
        "n_vi_clean": len(vi_clean),
        "n_en_clean": len(en_clean),
    }


def _rate(curve: dict, share: float) -> float:
    vals = [v["p_vi"] for k, v in curve.items() if v["share"] == share]
    return statistics.fmean(vals) if vals else math.nan


def analyse(rows: list[dict], ref: dict) -> dict:
    out = {}
    for name, fn in CANDIDATES.items():
        curve: dict = {}
        for share in SHARES:
            for construction in ("clean",) if share in (0.0, 1.0) else CONSTRUCTIONS:
                subset = [
                    r
                    for r in rows
                    if r["share"] == share and r["construction"] == construction
                ]
                if not subset:
                    continue
                margins = [decide(fn, r["vi"], r["en"], ref) for r in subset]
                finite = [m for m in margins if math.isfinite(m)]
                wins = sum(1 for m in margins if m > 0)
                lo, hi = wilson(wins, len(margins))
                curve[f"{share:g}|{construction}"] = {
                    "share": share,
                    "construction": construction,
                    "n": len(margins),
                    "p_vi": wins / len(margins),
                    "ci": [lo, hi],
                    "thin_share": (
                        sum(1 for m in finite if abs(m) < THIN) / len(finite)
                        if finite
                        else 0.0
                    ),
                    "median_margin": statistics.median(finite) if finite else math.nan,
                    "vi_empty": sum(1 for r in subset if not r["vi"]),
                    "en_empty": sum(1 for r in subset if not r["en"]),
                }
        need = math.nan
        for share in SHARES[1:]:
            if _rate(curve, share) >= 0.95:
                need = share
                break
        out[name] = {
            "clean_vi_recall": _rate(curve, 1.0),
            "clean_en_recall": 1.0 - _rate(curve, 0.0),
            "share_needed_for_95": need,
            "curve": curve,
        }
    return out


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gap-ms", type=int, default=80)
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args()

    vi_paths = sorted(str(p) for p in VI_WAVS.glob("*.wav"))[: args.pairs]
    en_paths = sorted(str(p) for p in EN_WAVS.glob("*.wav"))[: args.pairs]
    n = min(len(vi_paths), len(en_paths))
    if n < 5:
        raise SystemExit(f"need at least 5 pairs, found {n}")
    pairs = list(zip(vi_paths, en_paths, strict=True))

    tasks = []
    for share in SHARES:
        for construction in ("clean",) if share in (0.0, 1.0) else CONSTRUCTIONS:
            tasks.append((share, construction, pairs, args.gap_ms))

    print(
        f"{len(tasks)} conditions x {n} pairs = {len(tasks) * n} clips, 2 decodes each",
        flush=True,
    )
    rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
    ) as pool:
        for done, block in enumerate(pool.map(_run_condition, tasks), start=1):
            rows.extend(block["rows"])
            print(
                f"  {done}/{len(tasks)}  {block['share']:.0%} {block['construction']}",
                flush=True,
            )

    ref = fit_reference(rows)
    results = analyse(rows, ref)
    report = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": "decision quality only - no latency, no device timing",
        "note": (
            "argmax over statistics of per-token log-prob, both shipping checkpoints, "
            "synthetic splices of organiser clips. Spliced constructions pair different "
            "speakers, which the literature records as a confound: read these as an upper "
            "bound on separability, not as a simulation of code-switching."
        ),
        "pairs": n,
        "constants": {
            "shares": list(SHARES),
            "constructions": list(CONSTRUCTIONS),
            "top_fraction": TOP_FRACTION,
            "thin": THIN,
            "gap_ms": args.gap_ms,
        },
        "models": {"vi": str(VI_MODEL), "en": str(EN_MODEL)},
        "audio": {"vi": str(VI_WAVS), "en": str(EN_WAVS)},
        "commit": _commit(),
        "reference": ref,
        "results": results,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report), encoding="utf-8")

    print("\nstatistic   clean VI  clean EN  share needed for 95%")
    for name, r in results.items():
        need = r["share_needed_for_95"]
        label = "not reached" if math.isnan(need) else f"{need:.0%}"
        print(
            f"{name:<11} {r['clean_vi_recall']:>7.1%}  {r['clean_en_recall']:>7.1%}  {label}"
        )
    try:
        shown = args.out.resolve().relative_to(_ROOT)
    except ValueError:
        shown = args.out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
