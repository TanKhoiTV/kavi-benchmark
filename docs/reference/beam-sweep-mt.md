# Beam-width sweep — MT (formal rerun)

**Date:** 2026-10-24 · **Dataset:** mt_vi_en_eval_manifest.json (30 items) · **Candidate:** `opus-mt-vi-en-ct2-cpu` · **Repeats:** 3 per item after 1 discarded warmup pass(es) · **Beams:** 1, 2, 4, 5, 8

| beam | n | sentence-BLEU | corpus BLEU | chrF | p50 s | p95 s | median ratio | words/s | KV MB (derived) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 7.25 | 4.5 | 35.77 | 0.312 | 1.459 | 1.0 | 150.26 | 6.0 |
| 2 | 30 | 9.91 | 6.63 | 41.4 | 0.211 | 0.945 | 0.68 | 156.46 | 12.0 |
| 4 | 30 | 11.54 | 8.79 | 45.63 | 0.344 | 4.168 | 1.1 | 51.21 | 24.0 |
| 5 | 30 | 10.34 | 6.99 | 42.83 | 0.388 | 3.006 | 1.24 | 51.71 | 30.0 |
| 8 | 30 | 11.34 | 8.8 | 46.28 | 0.39 | 4.167 | 1.25 | 31.09 | 48.0 |

Corpus BLEU signature: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`.

## Verdict

The definitions disagree: sentence-BLEU peaks at beam=4 (11.54), corpus BLEU at beam=8 (8.8). Report both or report neither. beam=8 and beam=4 are **not separable on quality** (0.01 corpus BLEU, inside the 0.15 this sample can resolve); beam=4 is the cheaper of the two.

Repeats are what make the latency column citable: every figure above is a **median of 3 timed passes per item**, with the first 1 pass(es) discarded so framework warmup is not inside the sample.

## What this fixes

- **One sample per item → 3 plus warmup.** The archived sweep timed a single call with no warmup, which is why beam=4 beat beam=1 on 3 of its 30 items.
- **Per-sentence BLEU only → both definitions.** `bench/scorer.py` scores a single segment as a corpus, so its per-item figure weights a 3-word item like a 40-word one. The two columns disagree upstream of the peak, and naming the definition is the difference between a claim and a guess.
- **Beam width now recorded per row** (kavi-benchmark#142), so the run reproduces from its own output.
- **Model loaded once, reconfigured per beam** (kavi-benchmark#90 problem 4). `beam_size` is runtime config, not model identity.
- **KV cache computed, not "measured".** The archived RSS-delta column reported beam=8 holding less cache than beam=2, which is impossible: beam search keeps one cache per hypothesis. The derived column scales linearly with beam by construction.

## Monotonicity

57 of 120 adjacent-beam transitions have the wider beam finish faster. First instance: `vi-en-mt-1660: beam=1 1.371s > beam=2 0.913s`.

## What this still cannot claim

These are **host CPU** numbers. Whether beam=4 fits the 2.0 s turnaround gate is an on-device question (ADR-020, ADR-023), because the MT decoder runs on CPU with no NPU offload. Use the quality columns here to pick a beam, and the device measurement to accept it.

**Do not quote the p50 column as a turnaround figure.** It excludes audio capture, ASR, TTS, and every handoff; it is a laptop CPU decoding a cached model. The on-device half arrives from the app runner, and the encoder's HTP profile from Qualcomm AI Hub (kavi-android#31) — neither can be inferred from the table above.

Regenerate this file from a finished sweep without re-running it:

```bash
uv run python -m bench.beam_sweep --report-only     --out bench-results/beam-sweep-mt --report docs/reference/beam-sweep-mt.md
```
