# Beam-sweep host aggregate — Opus-MT vi→en

**Date:** 2026-09-28 **Dataset:** 30 MT items with `reference_text`, from `eval_manifest_v1.json` **Candidate:** `opus-mt-vi-en-ct2-cpu` — Helsinki-NLP Opus-MT vi→en via CTranslate2 int8, CPU **Status:** supporting evidence for open parameter #6 in [`docs/decisions/README.md`](../decisions/README.md). **Not a decision.** See ADR-020 — *pipeline concurrency* (`kavi-android`) for the deferral.

This file exists because the numbers quoted in ADR-020 need a source a reader can check. The raw per-item output is not one: it lives under `bench-results/`, which is gitignored (`.gitignore:44`) and therefore tracked on no branch. This is the reduced table that *is* tracked, plus the method that produces it.

> **Superseded for the quality question.** A formal rerun — [`beam-sweep-mt.md`](beam-sweep-mt.md) — replaces this table as evidence for the beam-width parameter: it measures three repeats per item after a discarded warmup pass, reports both BLEU definitions with a signature, records the beam width per row, and derives the KV-cache column instead of subtracting process RSS. The table below is kept as history, and because its latency column is the record of *why* singleshot timing was abandoned.

## Reproducing

```bash
uv run python -m scripts.summarise_beam_sweep \
  bench-results/archive/beam-sweep/fleurs/opus-mt-vi-en-ct2-cpu
```

> This archive tree is untracked and is not written by any code in this repo — `git grep -ni fluers` matches nothing tracked, so the directory is a hand-made scratch tree. It was originally spelled `fluers` and has been renamed to match the dataset. If you hold a copy with the old name, rename it before running the command above.

## Aggregate

| beam | n | sentence-BLEU | corpus BLEU | mean s | mean ratio | median s | median ratio | p95 s |
|---|---|---|---|---|---|---|---|---|
| 1 (greedy) | 30 | 6.22 | 3.98 | 0.558 | 1.00× | 0.475 | 1.00× | 1.248 |
| 2 | 30 | 7.81 | 5.39 | 0.911 | 1.63× | 0.813 | 1.71× | 1.878 |
| 4 | 30 | 8.80 | 6.58 | 1.396 | 2.50× | 0.616 | 1.30× | 3.238 |
| 5 | 30 | 9.01 | 6.08 | 1.686 | 3.02× | 0.861 | 1.81× | 3.482 |
| 8 | 30 | 8.97 | 6.42 | 2.489 | 4.46× | 1.536 | 3.23× | 6.560 |

**The two BLEU columns disagree, and not only in magnitude.** The sweep's own figure is a *mean of per-sentence* BLEU; `corpus BLEU` is SacreBLEU over the whole item set. Averaging per-sentence scores weights a 3-word item the same as a 40-word one. The practical consequence is that the ranking moves in the middle of the ladder:

- by sentence-BLEU the best is **beam=5** (9.01)
- by corpus BLEU the best is **beam=4** (6.58), with beam=5 *dropping* to 6.08 and beam=8 climbing back to 6.42 — non-monotonic where sentence-BLEU was flat

The published "beam=4 sweet spot" survives under corpus BLEU, but it is not the conclusion the sweep's own column supports. The margin is 0.16 over beam=8 and 0.50 over beam=5, on 30 items — small enough that beams 4, 5 and 8 are not separable on quality. Anything quoted from this sweep should name its BLEU definition.

## Why the latency column is not citable

Two independent defects, both visible in the aggregate above.

**1. Mean and median disagree, and the mean carries a long tail.** beam=4 has a median of 0.616 s but a p95 of 3.226 s and a max of 3.810 s. Its mean-based ratio (2.50×) is the figure the sweep published; its median-based ratio is 1.30×. The spread is wide enough at every beam that the ordering of the ratios depends on which statistic you pick — beam=2 is *slower* than beam=4 at the median, faster at the mean.

**2. Per-item latency is not monotonic in beam width.** Beam search increases decoder work monotonically *at a fixed output length*, so `b1 ≤ b2 ≤ b4 ≤ b5 ≤ b8` is the expected ordering. It is not guaranteed per item, though: beams can emit outputs of different lengths, and decoder steps scale with output length. It does not hold:

- **18 of 30 items (60%)** break the ordering
- across the four adjacent transitions, **24 pairs** have the wider beam finishing *faster* than the narrower one

Examples, all five beams, seconds:

| item | b1 | b2 | b4 | b5 | b8 |
|---|---|---|---|---|---|
| `vi-en-mt-1730` | 1.085 | 1.238 | **0.355** | 3.289 | 4.733 |
| `vi-en-mt-1675` | 0.840 | 1.339 | **0.472** | 0.607 | 1.511 |
| `vi-en-mt-1951` | 0.874 | 1.063 | **0.549** | 0.693 | 1.825 |

In all three, beam=4 is faster than **beam=1**. What happens next is not uniform: `vi-en-mt-1730` jumps roughly 9× at beam=5, while the other two rise by only about 1.3×.

**Root cause: one timing sample per item.** The records hold exactly one `latency_s` per item (30 items → 30 samples). `Candidate.run()` wraps a single `_infer()` call in `time.perf_counter()` — no repeats, no warmup discard. beam=1 min is 0.049 s against a 0.475 s median. That spread is consistent with the mix of item lengths in this set (3-word alongside 40-word items); it does not by itself show that framework warmup is inside the sample, which would need the first item's reading shown as anomalous. This is a separate defect from the memory-profiling issue raised in the sweep's own retrospective, and it is not fixed by a clean clock reading of a single noisy run.

## What this does and does not support

**Supports:** the *quality* side of the beam trade-off. BLEU rises materially from greedy to beam=4 under either definition (+2.58 sentence-BLEU, +2.60 corpus BLEU), and flattens or regresses beyond that. This is the one measurement in the sweep that survives scrutiny.

**Does not support:** any claim about whether beam=4 fits the latency budget. These are host-relative ratios from a laptop CPU. The 2.0 s turnaround gate in ADR-020 is an on-device number, and the on-device MT runner is still a stub. The gap between 2.50× and 1.30× above is enough to flip the answer either way, which is precisely why the host column cannot stand in for the device measurement.

## Related

- ADR-020 — *pipeline concurrency* (`kavi-android`) — the deferral this aggregate supports; open parameter #6
- [`scripts/summarise_beam_sweep.py`](../../scripts/summarise_beam_sweep.py) — regenerates this table
- `feat/beam-sweep-mt` — the branch holding the harness and the per-model reports. Note the Opus-MT report there is dated 2025-07-26; the branch's own history puts it at 2026-07-27, and its retrospective table carries the same year error.
- The Opus-MT report there also averages per-sentence BLEU, which is what section *The two BLEU columns disagree* is about. Its `beam=4` verdict holds under corpus BLEU, but on a 0.16 margin over beam=8 (0.50 over beam=5) across 30 items.
