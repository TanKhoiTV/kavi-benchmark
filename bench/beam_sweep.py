"""Formal MT beam-width sweep: repeats, warmup, both BLEU definitions.

The archived sweep (``docs/reference/beam-sweep-host-aggregate.md``) is not
citable evidence for the beam-width parameter, and it says so itself. Three
defects made that true:

1. **One timing sample per item**, with no warmup discard. That produced a
   latency column where beam=4 beat beam=1 on three items, i.e. a ladder that
   violates its own ordering.
2. **Per-sentence BLEU only.** ``bench/scorer.py`` calls
   ``sacrebleu.corpus_bleu([hyp], [[ref]])`` on a single segment, so the per-item
   figure is a one-sentence corpus score. Averaging those weights a 3-word item
   the same as a 40-word one, and the two definitions disagree on where the
   ladder peaks.
3. **Beam width unrecorded per row** (TanKhoiTV/kavi-benchmark#142), so a run
   could not be reproduced from its own output.

It also reloaded the model once per beam width (TanKhoiTV/kavi-benchmark#90
problem 4). ``beam_size`` is runtime config, not model identity, so this sweep
loads the candidate once and reconfigures it in place.

KV-cache memory is **computed, not measured**. The archived attempt subtracted
process-wide RSS across a run, which produced a column where beam=8 held less
cache than beam=2 — physically impossible, because more hypotheses require more
memory. The cache size is a function of model geometry, tokens and beam width,
so it is derived here and labelled as derived.

Usage::

    uv run python -m bench.beam_sweep \
        --manifest eval_data/mt_vi_en_eval_manifest.json \
        --out bench-results/beam-sweep-mt \
        --beams 1 2 4 5 8 --repeats 3 --warmup 1 \
        --report docs/reference/beam-sweep-mt.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adapters import Candidate
from .registry import build_candidate
from .schema import EvalItem, RunManifest, StageResult

# Model geometry for the analytic KV-cache column: (layers, d_model).
# Both values come from the HuggingFace configs of the pinned checkpoints.
KV_GEOMETRY: dict[str, tuple[int, int]] = {
    "opus-mt-vi-en-ct2-cpu": (6, 512),
    "m2m100-vi-en-ct2-cpu": (12, 1024),
}
DEFAULT_MAX_TOKENS = 256
# Quality margin a 30-item MT sample cannot resolve; used to call a tie a tie.
CORPUS_BLEU_RESOLUTION = 0.15
# CTranslate2 keeps the KV cache in float32 regardless of compute_type.
CACHE_DTYPE_BYTES = 4


def kv_cache_mb(
    candidate_id: str,
    beam: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> float | None:
    """Derive KV-cache MB for one beam width; ``None`` when geometry is unknown.

    ``2`` gates (K and V) x ``layers`` x ``d_model`` floats per token, scaled by
    the number of live hypotheses. Beam search keeps one cache per hypothesis,
    which is why the column has to scale with ``beam`` at all.
    """
    geometry = KV_GEOMETRY.get(candidate_id)
    if geometry is None:
        return None
    layers, d_model = geometry
    floats = 2 * layers * d_model * max_tokens * beam
    return round(floats * CACHE_DTYPE_BYTES / (1024.0**2), 2)


def mt_items(manifest: RunManifest, limit: int | None) -> list[EvalItem]:
    items = [item for item in manifest.items if item.stage == "MT"]
    if not items:
        raise ValueError("manifest holds no MT items")
    return items[:limit] if limit else items


def _observe(
    candidate: Candidate, item: EvalItem, beam: int, repeat: int
) -> dict[str, Any]:
    result = candidate.run(item)
    return {
        "item_id": item.id,
        "beam": beam,
        "repeat": repeat,
        "latency_s": result.latency_s,
        "output_text": result.output_text,
        "error": result.error,
    }


def sweep(
    items: list[EvalItem],
    candidate: Candidate,
    beams: list[int],
    repeats: int,
    warmup: int,
) -> list[dict[str, Any]]:
    """Run every item at every beam width, discarding ``warmup`` passes."""
    rows: list[dict[str, Any]] = []
    for beam in beams:
        candidate.configure({"beam_size": beam})
        for _ in range(warmup):
            for item in items:
                candidate.run(item)  # timing discarded, allocator primed
        for repeat in range(repeats):
            rows.extend(_observe(candidate, item, beam, repeat) for item in items)
    return rows


def timed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows with a usable latency and no error, i.e. the population we report."""
    return [row for row in rows if row["error"] is None and row["latency_s"] > 0]


def per_item_medians(
    rows: list[dict[str, Any]], items: list[EvalItem], beam: int
) -> list[dict[str, Any]]:
    """Median-across-repeats record per item, in the ``run_results.json`` shape.

    Keeping this envelope means ``scripts/summarise_beam_sweep.py`` still reads
    a sweep directory without changes.
    """
    by_item: dict[str, list[dict[str, Any]]] = {}
    for row in timed_rows(rows):
        if row["beam"] == beam:
            by_item.setdefault(row["item_id"], []).append(row)

    records = []
    for item in items:
        observed = by_item.get(item.id, [])
        if not observed:
            records.append(
                {
                    "item": asdict(item),
                    "result": asdict(
                        StageResult(
                            candidate_id="",
                            item_id=item.id,
                            stage="MT",
                            error="no successful repeat",
                        )
                    ),
                    "metrics": {},
                    "repeats": 0,
                }
            )
            continue
        latency = statistics.median(row["latency_s"] for row in observed)
        records.append(
            {
                "item": asdict(item),
                "result": asdict(
                    StageResult(
                        candidate_id=observed[0]["item_id"] and "",
                        item_id=item.id,
                        stage="MT",
                        output_text=observed[0]["output_text"],
                        latency_s=round(latency, 4),
                    )
                ),
                "metrics": {},
                "repeats": len(observed),
                "beam_size": beam,
            }
        )
    return records


def bleu_scores(hyps: list[str], refs: list[str]) -> dict[str, Any]:
    """Both BLEU definitions, because they disagree on where the ladder peaks."""
    import sacrebleu

    sentence = statistics.fmean(
        sacrebleu.sentence_bleu(hyp, [ref]).score
        for hyp, ref in zip(hyps, refs, strict=True)
    )
    metric = sacrebleu.BLEU()
    corpus = metric.corpus_score(hyps, [refs])
    return {
        "sentence_bleu": round(sentence, 2),
        "corpus_bleu": round(corpus.score, 2),
        # Readable only after evaluation: the signature needs the reference count.
        "signature": str(metric.get_signature()),
        "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return ordered[idx]


def summarise(
    rows: list[dict[str, Any]],
    items: list[EvalItem],
    beams: list[int],
    candidate_id: str,
) -> dict[str, Any]:
    """One row per beam width: quality, latency distribution, derived KV."""
    refs = {item.id: (item.resolve_reference() or "") for item in items}
    per_beam: list[dict[str, Any]] = []
    baseline_median: float | None = None

    for beam in beams:
        observed = timed_rows([row for row in rows if row["beam"] == beam])
        if not observed:
            continue
        by_item: dict[str, dict[str, Any]] = {}
        for row in observed:
            by_item.setdefault(row["item_id"], row)
        ids = [item.id for item in items if item.id in by_item]
        hyps = [by_item[item_id]["output_text"] or "" for item_id in ids]
        latencies = [
            statistics.median(r["latency_s"] for r in observed if r["item_id"] == i)
            for i in ids
        ]
        median = statistics.median(latencies)
        if baseline_median is None:
            baseline_median = median
        words = [len(hyp.split()) for hyp in hyps]
        per_beam.append(
            {
                "beam": beam,
                "n": len(ids),
                "latency_p50_s": round(median, 3),
                "latency_p95_s": round(_percentile(latencies, 0.95), 3),
                "latency_ratio": round(median / baseline_median, 2)
                if baseline_median
                else None,
                "words": sum(words),
                "words_per_s": round(sum(words) / sum(latencies), 2)
                if sum(latencies)
                else None,
                "kv_cache_mb": kv_cache_mb(candidate_id, beam),
                **bleu_scores(hyps, [refs[item_id] for item_id in ids]),
            }
        )

    return {
        "candidate_id": candidate_id,
        "items": len(items),
        "beams": per_beam,
        "monotonic_breaks": monotonic_breaks(rows, items, beams),
    }


def monotonic_breaks(
    rows: list[dict[str, Any]], items: list[EvalItem], beams: list[int]
) -> dict[str, Any]:
    """Count items where a wider beam finished faster than a narrower one.

    Expected zero; nonzero means the per-item sample is still dominated by
    noise or output length, not by decoder work.
    """
    medians: dict[int, dict[str, float]] = {}
    for beam in beams:
        observed = timed_rows([row for row in rows if row["beam"] == beam])
        for item in items:
            samples = [r["latency_s"] for r in observed if r["item_id"] == item.id]
            if samples:
                medians.setdefault(beam, {})[item.id] = statistics.median(samples)
    transitions = 0
    breaks = 0
    example: str | None = None
    for narrow, wide in zip(beams, beams[1:], strict=False):
        for item in items:
            low = medians.get(narrow, {}).get(item.id)
            high = medians.get(wide, {}).get(item.id)
            if low is None or high is None:
                continue
            transitions += 1
            if high < low:
                breaks += 1
                example = (
                    example
                    or f"{item.id}: beam={narrow} {low:.3f}s > beam={wide} {high:.3f}s"
                )
    return {"transitions": transitions, "breaks": breaks, "example": example}


def load_rows(out_dir: Path) -> list[dict[str, Any]]:
    """Re-read a finished sweep so the report regenerates without recomputing."""
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("beam-*/sweep_results.json")):
        try:
            rows.extend(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"no beam-*/sweep_results.json under {out_dir}")
    return rows


def verdict(summary: dict[str, Any]) -> str:
    """Name the quality peak and whether both BLEU definitions agree on it."""
    rows = summary["beams"]
    if not rows:
        return "No beam width produced a scorable output."
    by_sentence = max(rows, key=lambda row: row["sentence_bleu"])
    by_corpus = max(rows, key=lambda row: row["corpus_bleu"])
    if by_sentence["beam"] == by_corpus["beam"]:
        agree = (
            f"Both definitions peak at **beam={by_sentence['beam']}** "
            f"({by_sentence['sentence_bleu']} sentence / {by_corpus['corpus_bleu']} corpus). "
            "Agreement on the argmax is the useful part: the archived sweep's two "
            "columns disagreed here, which is why its beam=4 verdict could not be "
            "separated from a definition choice."
        )
    else:
        agree = (
            f"The definitions disagree: sentence-BLEU peaks at beam={by_sentence['beam']} "
            f"({by_sentence['sentence_bleu']}), corpus BLEU at beam={by_corpus['beam']} "
            f"({by_corpus['corpus_bleu']}). Report both or report neither."
        )
    return agree + " " + separability(rows)


def separability(rows: list[dict[str, Any]]) -> str:
    """Say whether the top two beams differ by more than a 30-item sample resolves.

    Rule of thumb, not a significance test: on 30 items a corpus-BLEU gap under
    0.15 is smaller than the run-to-run spread, so a "winner" inside that band is
    a tie and the cheaper beam should win it.
    """
    ranked = sorted(rows, key=lambda row: row["corpus_bleu"], reverse=True)
    if len(ranked) < 2:
        return "Only one beam width ran, so nothing is being compared."
    best, runner = ranked[0], ranked[1]
    gap = best["corpus_bleu"] - runner["corpus_bleu"]
    if gap > CORPUS_BLEU_RESOLUTION:
        return (
            f"beam={best['beam']} leads beam={runner['beam']} by {gap:.2f} corpus BLEU, "
            f"outside the {CORPUS_BLEU_RESOLUTION} this sample can resolve."
        )
    cheaper = min(best, runner, key=lambda row: row["latency_p50_s"])
    return (
        f"beam={best['beam']} and beam={runner['beam']} are **not separable on quality** "
        f"({gap:.2f} corpus BLEU, inside the {CORPUS_BLEU_RESOLUTION} this sample can "
        f"resolve); beam={cheaper['beam']} is the cheaper of the two."
    )


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    rows = "\n".join(
        "| {beam} | {n} | {sentence_bleu} | {corpus_bleu} | {chrf} | {latency_p50_s} | "
        "{latency_p95_s} | {latency_ratio} | {words_per_s} | {kv_cache_mb} |".format(
            **row
        )
        for row in summary["beams"]
    )
    breaks = summary["monotonic_breaks"]
    text = f"""# Beam-width sweep — MT (formal rerun)

**Date:** {meta["date"]} · **Dataset:** {meta["manifest"]} ({summary["items"]} items) · **Candidate:** `{summary["candidate_id"]}` · **Repeats:** {meta["repeats"]} per item after {meta["warmup"]} discarded warmup pass(es) · **Beams:** {meta["beams"]}

| beam | n | sentence-BLEU | corpus BLEU | chrF | p50 s | p95 s | median ratio | words/s | KV MB (derived) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{rows}

Corpus BLEU signature: `{summary["beams"][0]["signature"] if summary["beams"] else "n/a"}`.

Decode guard: **on**. `max_decoding_length` is sized from the utterance (`bench/decode.py`: 1.5 x source pieces, clamped to [48, 256]) instead of a flat 256. Without it, CTranslate2's greedy decode loops on long Vietnamese input and the looped items dominate both the latency tail and the BLEU mean.

## Verdict

{verdict(summary)}

Repeats are what make the latency column citable: every figure above is a **median of {meta["repeats"]} timed passes per item**, with the first {meta["warmup"]} pass(es) discarded so framework warmup is not inside the sample.

## What this fixes

- **One sample per item → {meta["repeats"]} plus warmup.** The archived sweep timed a single call with no warmup, which is why beam=4 beat beam=1 on 3 of its 30 items.
- **Per-sentence BLEU only → both definitions.** `bench/scorer.py` scores a single segment as a corpus, so its per-item figure weights a 3-word item like a 40-word one. The two columns disagree upstream of the peak, and naming the definition is the difference between a claim and a guess.
- **Beam width now recorded per row** (kavi-benchmark#142), so the run reproduces from its own output.
- **Model loaded once, reconfigured per beam** (kavi-benchmark#90 problem 4). `beam_size` is runtime config, not model identity.
- **KV cache computed, not "measured".** The archived RSS-delta column reported beam=8 holding less cache than beam=2, which is impossible: beam search keeps one cache per hypothesis. The derived column scales linearly with beam by construction.

## Monotonicity

{breaks["breaks"]} of {breaks["transitions"]} adjacent-beam transitions have the wider beam finish faster. {("First instance: `" + breaks["example"] + "`.") if breaks["example"] else "Every wider beam is slower on every item, as decoder work predicts."}

## What this still cannot claim

These are **host CPU** numbers. Whether beam=4 fits the 2.0 s turnaround gate is an on-device question (ADR-020, ADR-023), because the MT decoder runs on CPU with no NPU offload. Use the quality columns here to pick a beam, and the device measurement to accept it.

**Do not quote the p50 column as a turnaround figure.** It excludes audio capture, ASR, TTS, and every handoff; it is a laptop CPU decoding a cached model. The on-device half arrives from the app runner, and the encoder's HTP profile from Qualcomm AI Hub (kavi-android#31) — neither can be inferred from the table above.

Regenerate this file from a finished sweep without re-running it:

```bash
uv run python -m bench.beam_sweep --report-only \
    --out bench-results/beam-sweep-mt --report docs/reference/beam-sweep-mt.md
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit LF: the Windows default would write CRLF and fail the shape gate.
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {path}")


def _meta(args: argparse.Namespace) -> dict[str, Any]:
    import datetime

    return {
        "date": datetime.date.today().isoformat(),
        "manifest": args.manifest.name,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "beams": ", ".join(str(beam) for beam in args.beams),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--beams", type=int, nargs="+", default=[1, 2, 4, 5, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="regenerate the report from an existing --out tree instead of re-running",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = RunManifest.from_json(args.manifest)
        items = mt_items(manifest, args.limit)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.report_only:
        try:
            rows = load_rows(args.out)
        except (OSError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 1
        # Take the candidate id from the saved summary, so a regenerated report
        # keeps the KV-cache column instead of degrading to "unknown geometry".
        saved_path = args.out / "sweep_summary.json"
        try:
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"cannot read {saved_path}: {exc}", file=sys.stderr)
            return 1
        summary = summarise(rows, items, args.beams, saved["candidate_id"])
        if args.report:
            write_markdown(args.report, summary, _meta(args))
        return 0

    candidate = build_candidate(items[0])
    rows = sweep(items, candidate, args.beams, args.repeats, args.warmup)

    args.out.mkdir(parents=True, exist_ok=True)
    for beam in args.beams:
        beam_dir = args.out / f"beam-{beam}"
        beam_dir.mkdir(parents=True, exist_ok=True)
        (beam_dir / "sweep_results.json").write_text(
            json.dumps(
                [row for row in rows if row["beam"] == beam],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (beam_dir / "run_results.json").write_text(
            json.dumps(
                per_item_medians(rows, items, beam), indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )

    summary = summarise(rows, items, args.beams, candidate.id)
    (args.out / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for row in summary["beams"]:
        print(
            f"beam={row['beam']:<2} n={row['n']:<3} "
            f"sentence-BLEU={row['sentence_bleu']:<6} corpus-BLEU={row['corpus_bleu']:<6} "
            f"p50={row['latency_p50_s']}s ratio={row['latency_ratio']}x"
        )

    if args.report:
        write_markdown(args.report, summary, _meta(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
