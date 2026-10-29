"""How much Vietnamese audio does the confidence comparison need before it says VI?

`lid_codeswitch.py` showed that the margin does not degrade on mixed-language
audio and settled on three coarse ratios. This sweep replaces them with a
sensitivity curve, because the question that decides the detection policy is
quantitative: at what Vietnamese share does the comparison become reliable, and
does an asymmetric prior buy the reliability that reality requires?

Two constructions, and the difference between them is the point:

* **contiguous** - the minority language appears as one block. This is what the
  first probe used, and it is harsher than real speech: a 70%-English stretch is
  a language change, not an embedded word.
* **embedded** - the minority appears as three short spans spread *inside* the
  majority clip, which is what "cái này ship đi được không?" actually sounds
  like: Vietnamese prosody running underneath English words.

The δ axis turns the policy question into a measured trade. Adding δ to the
Vietnamese side of the margin biases the decision toward VI; the artifact records
the margin of every clip, so the curve can be recomputed for any δ without
re-decoding.

Every number here is **decision quality** - which language the comparison picks.
None of it is a latency claim: turnaround has to be measured on the phone, in the
app, and a host decode time is a diagnostic that says nothing about the device.
That is why the artifact records no timing at all.

Runs on CPU on purpose. `ys_log_probs` is the int8 CPU signal the phone actually
produces, and sherpa-onnx's `provider` accepts only cpu/cuda/coreml, so an
OpenVINO or DirectML run on the Arc GPU would measure different numerics - a
different signal, not a faster one. Speed comes from worker processes instead,
which is a throughput change and not a measurement change.

Usage:

    uv run --with sherpa-onnx==2.0.1 --with sherpa-onnx-core==2.0.1 \
        python -m bench.lid_share_sweep --pairs 40 --workers 4
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .lid_codeswitch import (
    EN_MODEL,
    EN_WAVS,
    THIN,
    VI_MODEL,
    VI_WAVS,
    load_recognizers,
    margins_for,
    read_wav,
)

_ROOT = Path(__file__).resolve().parents[1]

#: Vietnamese share of the audio, by duration. Dense where the decision flips.
SHARES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70)
CONSTRUCTIONS = ("contiguous", "embedded")
EMBED_SPANS = 3

#: One recogniser pair per worker; filled by `_init_worker`.
_WORKER: dict = {}


@dataclass(frozen=True)
class Condition:
    vi_share: float
    construction: str

    @property
    def label(self) -> str:
        return f"{self.vi_share:.0%} VI {self.construction}"


def build(
    vi: np.ndarray,
    en: np.ndarray,
    sr: int,
    *,
    vi_share: float,
    construction: str,
    gap_ms: int = 80,
    spans: int = EMBED_SPANS,
) -> np.ndarray:
    """Assemble one mixed clip whose Vietnamese share is `vi_share` by duration."""
    gap = np.zeros(int(sr * gap_ms / 1000), dtype=np.float32)
    total = min(len(vi) / vi_share, len(en) / max(1e-9, 1.0 - vi_share))
    vi_take = max(1, int(round(vi_share * total)))
    en_take = max(1, int(round((1.0 - vi_share) * total)))

    if construction == "contiguous":
        return np.concatenate([vi[:vi_take], gap, en[:en_take]])

    if vi_share >= 0.5:
        majority, minority, majority_take, minority_take = vi, en, vi_take, en_take
    else:
        majority, minority, majority_take, minority_take = en, vi, en_take, vi_take

    span_len = max(1, minority_take // spans)
    seg_len = max(1, majority_take // (spans + 1))
    pieces: list[np.ndarray] = []
    maj_at = 0
    min_at = 0
    for _ in range(spans):
        pieces.append(majority[maj_at : maj_at + seg_len])
        maj_at += seg_len
        pieces.append(gap)
        pieces.append(minority[min_at : min_at + span_len])
        min_at += span_len
        pieces.append(gap)
    pieces.append(majority[maj_at : maj_at + seg_len])
    return np.concatenate(pieces)


def _init_worker() -> None:
    vi_rec, en_rec = load_recognizers()
    _WORKER["vi"] = vi_rec
    _WORKER["en"] = en_rec


def _run_condition(task: tuple[Condition, list[str], list[str], int]) -> dict:
    condition, vi_paths, en_paths, gap_ms = task
    rows = []
    for vi_path, en_path in zip(vi_paths, en_paths, strict=False):
        vi_samples, sr = read_wav(Path(vi_path))
        en_samples, _ = read_wav(Path(en_path))
        mixed = build(
            vi_samples,
            en_samples,
            sr,
            vi_share=condition.vi_share,
            construction=condition.construction,
            gap_ms=gap_ms,
        )
        m = margins_for(_WORKER["vi"], _WORKER["en"], mixed, sr)
        rows.append(
            {
                "vi_mean": m.vi_mean,
                "en_mean": m.en_mean,
                "margin": m.margin,
                "vi_tokens": m.vi_tokens,
                "en_tokens": m.en_tokens,
                "total_seconds": len(mixed) / sr,
                "vi_seconds": condition.vi_share * len(mixed) / sr,
            }
        )
    return {
        "condition": condition.label,
        "vi_share": condition.vi_share,
        "construction": condition.construction,
        "rows": rows,
    }


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion, which is honest at n=40 and p near 0 or 1."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def p_vi(rows: list[dict], delta: float = 0.0) -> float:
    """Share of clips the comparison sends to Vietnamese, with an optional VI bias.

    A clip where the Vietnamese model produced tokens and the English model
    produced none counts as a Vietnamese win outright: an empty hypothesis is
    evidence, not a missing measurement.
    """
    wins = 0
    counted = 0
    for row in rows:
        counted += 1
        if row["margin"] is not None:
            wins += 1 if row["margin"] + delta >= 0 else 0
        elif row["vi_mean"] is not None:
            wins += 1
    return wins / counted if counted else float("nan")


def _clean_rows(vi_rec, en_rec, paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        samples, sr = read_wav(Path(path))
        m = margins_for(vi_rec, en_rec, samples, sr)
        rows.append(
            {
                "margin": m.margin,
                "vi_mean": m.vi_mean,
                "en_mean": m.en_mean,
                "vi_tokens": m.vi_tokens,
                "en_tokens": m.en_tokens,
            }
        )
    return rows


def analyse(results: list[dict], clean: dict[str, list[dict]]) -> dict:
    """The sensitivity curve, and the cost of a Vietnamese bias."""
    curve = []
    for res in sorted(results, key=lambda r: (r["construction"], r["vi_share"])):
        share = p_vi(res["rows"])
        n = len(res["rows"])
        lo, hi = wilson(int(round(share * n)), n)
        curve.append(
            {
                "condition": res["condition"],
                "vi_share": res["vi_share"],
                "construction": res["construction"],
                "p_vi": share,
                "p_vi_ci95": [lo, hi],
                "thin_share": float(
                    np.mean(
                        [
                            row["margin"] is None or abs(row["margin"]) < THIN
                            for row in res["rows"]
                        ]
                    )
                ),
            }
        )

    # The realistic case for this pair: Vietnamese majority with English embedded.
    realistic = [
        row
        for res in results
        if res["construction"] == "embedded" and res["vi_share"] >= 0.5
        for row in res["rows"]
    ]
    roc = []
    for delta in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        roc.append(
            {
                "delta": delta,
                "vi_recall_clean": p_vi(clean["vi"], delta),
                "false_vi_clean_en": p_vi(clean["en"], delta),
                "p_vi_realistic": p_vi(realistic, delta),
            }
        )
    return {"curve": curve, "bias_roc": roc}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gap-ms", type=int, default=80)
    parser.add_argument("--out", default="bench-results/lid-share-sweep/report.json")
    args = parser.parse_args()

    vi_paths = sorted(glob.glob(str(VI_WAVS / "*.wav")))[: args.pairs]
    en_paths = sorted(glob.glob(str(EN_WAVS / "*.wav")))[: args.pairs]
    if not vi_paths or not en_paths:
        raise FileNotFoundError(
            "organizer wavs not staged under eval_data/organizer/ - the calibration "
            "corpus is gitignored and not redistributed"
        )
    if not VI_MODEL.exists() or not EN_MODEL.exists():
        raise FileNotFoundError("ASR checkpoints not staged; see lid_codeswitch.py")

    conditions = [
        Condition(share, construction)
        for construction in CONSTRUCTIONS
        for share in SHARES
    ]
    print(
        f"{len(conditions)} conditions x {len(vi_paths)} pairs, {args.workers} workers",
        flush=True,
    )

    vi_rec, en_rec = load_recognizers()
    clean = {
        "vi": _clean_rows(vi_rec, en_rec, vi_paths),
        "en": _clean_rows(vi_rec, en_rec, en_paths),
    }
    print(
        f"clean baselines: VI {p_vi(clean['vi']):.0%} correct, "
        f"EN {1 - p_vi(clean['en']):.0%} correct",
        flush=True,
    )

    tasks = [(c, vi_paths, en_paths, args.gap_ms) for c in conditions]
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
    ) as pool:
        for i, res in enumerate(pool.map(_run_condition, tasks), start=1):
            results.append(res)
            print(
                f"  [{i}/{len(tasks)}] {res['condition']:24} "
                f"P(VI)={p_vi(res['rows']):.0%}",
                flush=True,
            )

    analysis = analyse(results, clean)

    print(f"\n{'condition':26} {'P(VI)':>6} {'95% CI':>14} {'thin':>6}")
    for row in analysis["curve"]:
        lo, hi = row["p_vi_ci95"]
        print(
            f"{row['condition']:26} {row['p_vi']:>6.0%} "
            f"{f'[{lo:.0%}, {hi:.0%}]':>14} {row['thin_share']:>6.0%}"
        )

    print(
        f"\n{'delta':>6} {'VI recall (clean)':>18} {'false VI (clean EN)':>20} "
        f"{'P(VI) realistic':>16}"
    )
    for row in analysis["bias_roc"]:
        print(
            f"{row['delta']:>6.2f} {row['vi_recall_clean']:>18.0%} "
            f"{row['false_vi_clean_en']:>20.0%} {row['p_vi_realistic']:>16.0%}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "measured_on": str(date.today()),
                "provenance": {
                    "vi_checkpoint": str(VI_MODEL.relative_to(_ROOT)),
                    "en_checkpoint": str(EN_MODEL.relative_to(_ROOT)),
                    "signal": "mean per-token log-probability (ys_log_probs), sherpa-onnx 2.0.1, CPU",
                    "why_cpu": "the shipping numerics; provider whitelist is cpu/cuda/coreml",
                    "shares": list(SHARES),
                    "constructions": list(CONSTRUCTIONS),
                    "embed_spans": EMBED_SPANS,
                    "gap_ms": args.gap_ms,
                    "pairs_per_condition": args.pairs,
                    "thin_threshold": THIN,
                    "synthetic": "spliced, not naturally code-switched speech",
                },
                "clean": {
                    "vi_p_vi": p_vi(clean["vi"]),
                    "en_false_vi": p_vi(clean["en"]),
                    "n_vi": len(clean["vi"]),
                    "n_en": len(clean["en"]),
                },
                "analysis": analysis,
                "raw": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nartifact -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
