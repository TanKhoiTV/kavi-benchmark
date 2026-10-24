# Beam-width sweep — MT (formal rerun)

**Date:** 2026-10-24 · **Dataset:** mt_vi_en_eval_manifest.json (30 items) · **Candidate:** `opus-mt-vi-en-ct2-cpu` · **Repeats:** 3 per item after 1 discarded warmup pass(es) · **Beams:** 1, 2, 4, 5, 8

| beam | n | sentence-BLEU | corpus BLEU | chrF | p50 s | p95 s | median ratio | words/s | KV MB (derived) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 10.37 | 10.08 | 47.56 | 0.188 | 0.327 | 1.0 | 202.45 | 6.0 |
| 2 | 30 | 11.16 | 10.72 | 48.71 | 0.449 | 0.667 | 2.39 | 89.59 | 12.0 |
| 4 | 30 | 12.45 | 11.91 | 50.16 | 0.549 | 1.007 | 2.92 | 65.13 | 24.0 |
| 5 | 30 | 11.77 | 11.06 | 49.19 | 0.36 | 0.652 | 1.92 | 93.28 | 30.0 |
| 8 | 30 | 11.74 | 11.17 | 49.39 | 0.426 | 0.873 | 2.26 | 76.38 | 48.0 |

Corpus BLEU signature: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`.

Decode guard: **on**. `max_decoding_length` is sized from the utterance (`bench/decode.py`: 1.5 x source pieces, clamped to [48, 256]) instead of a flat 256. Without it, CTranslate2's greedy decode loops on long Vietnamese input and the looped items dominate both the latency tail and the BLEU mean.

## Verdict

Both definitions peak at **beam=4** (12.45 sentence / 11.91 corpus). Agreement on the argmax is the useful part: the archived sweep's two columns disagreed here, which is why its beam=4 verdict could not be separated from a definition choice. beam=4 leads beam=8 by 0.74 corpus BLEU, outside the 0.15 this sample can resolve.

Repeats are what make the latency column citable: every figure above is a **median of 3 timed passes per item**, with the first 1 pass(es) discarded so framework warmup is not inside the sample.

## What this fixes

- **One sample per item → 3 plus warmup.** The archived sweep timed a single call with no warmup, which is why beam=4 beat beam=1 on 3 of its 30 items.
- **Per-sentence BLEU only → both definitions.** `bench/scorer.py` scores a single segment as a corpus, so its per-item figure weights a 3-word item like a 40-word one. The two columns disagree upstream of the peak, and naming the definition is the difference between a claim and a guess.
- **Beam width now recorded per row** (kavi-benchmark#142), so the run reproduces from its own output.
- **Model loaded once, reconfigured per beam** (kavi-benchmark#90 problem 4). `beam_size` is runtime config, not model identity.
- **KV cache computed, not "measured".** The archived RSS-delta column reported beam=8 holding less cache than beam=2, which is impossible: beam search keeps one cache per hypothesis. The derived column scales linearly with beam by construction.

## Monotonicity

36 of 120 adjacent-beam transitions have the wider beam finish faster. First instance: `vi-en-mt-1670: beam=1 0.485s > beam=2 0.453s`.

## What this still cannot claim

These are **host CPU** numbers. Whether beam=4 fits the 2.0 s turnaround gate is an on-device question (ADR-020, ADR-023), because the MT decoder runs on CPU with no NPU offload. Use the quality columns here to pick a beam, and the device measurement to accept it.

**Do not quote the p50 column as a turnaround figure.** It excludes audio capture, ASR, TTS, and every handoff; it is a laptop CPU decoding a cached model. The on-device half arrives from the app runner, and the encoder's HTP profile from Qualcomm AI Hub (kavi-android#31) — neither can be inferred from the table above.

Regenerate this file from a finished sweep without re-running it:

```bash
uv run python -m bench.beam_sweep --report-only     --out bench-results/beam-sweep-mt --report docs/reference/beam-sweep-mt.md
```
