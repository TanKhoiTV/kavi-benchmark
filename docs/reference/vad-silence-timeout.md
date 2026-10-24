# VAD silence timeout — measured against real pause distributions

**Date:** 2026-10-24 · **Corpus:** organizer sample set, 100 clips per direction (16 kHz mono, VI mean 3.34 s / EN mean 5.96 s) · **Segmenter:** silero-vad, threshold 0.5, `speech_pad_ms` 0, `min_silence_duration_ms` 100 · **Measured:** `bench/vad_pause_stats.py`

`vad_speech_timeout_ms: 500` is the trailing silence required before endpointing fires, so it sits inside the 2.0 s turnaround budget. The tempting move is to cut it to 300 ms and bank 200 ms — as much as the entire beam=1 to beam=2 decode delta — for no model change. **That move is not available.** The measurement below is why.

| metric | Vie-Eng | Eng-Vie |
| --- | --- | --- |
| clips | 100 | 100 |
| clips containing pauses | 20 | 55 |
| pauses measured | 21 | 103 |
| p50 | 256 ms | 320 ms |
| p95 | 544 ms | 1 184 ms |
| p99 | 672 ms | 1 696 ms |
| max | 672 ms | 1 792 ms |
| mean | 282 ms | 416 ms |
| pauses ≥ 200 ms | 16 (76.2%) | 80 (77.7%) |
| pauses ≥ 250 ms | 11 (52.4%) | 67 (65.0%) |
| **pauses ≥ 300 ms** | **6 (28.6%)** | **52 (50.5%)** |
| pauses ≥ 400 ms | 3 (14.3%) | 38 (36.9%) |
| pauses ≥ 500 ms | 2 (9.5%) | 24 (23.3%) |

`min_silence_duration_ms: 100` is the measurement floor: gaps shorter than that are not separable from ordinary inter-word spacing, so they are excluded rather than counted as zero. It does not affect the ≥300 ms and ≥500 ms counts, which are exact.

## Verdict

**300 ms is unsafe, and 500 ms is already marginal for English.**

A timeout fires when silence exceeds it, so every pause longer than the timeout ends the turn early and the utterance is translated as fragments. At 300 ms that happens on **28.6% of Vietnamese pauses and 50.5% of English pauses** — the median English pause (320 ms) is itself longer than the proposed timeout.

The current 500 ms is not comfortable either: **23.3% of English pauses exceed it**, so a quarter of English sample utterances would already be split mid-sentence by today's setting. That is a finding in its own right, and it argues for raising the timeout rather than lowering it.

## Why this contradicts the earlier proposal

Cutting 500 → 300 ms was proposed as a free 200 ms, on the arithmetic that the timeout is pure budget consumption. The arithmetic is right and the premise was wrong: the timeout is not slack, it is headroom over the pause distribution, and 300 ms sits *inside* that distribution for both directions.

The 200 ms therefore has a real price — truncated translations on half of English utterances — paid in criterion A quality (45% of the score) to buy latency (20%). That trade is not worth taking, and it is no longer necessary: the decoder measurement (kavi-android#41) projects 80–150 ms for the entire MT stage against 507 ms today, which returns several times the 200 ms without touching endpointing.

## What this does not establish

The sample corpus is **clean studio-like speech** (frame SNR ~50 dB, no clipping), which is the optimistic case for pause detection: noise generally inserts *more* energy in gaps and makes them harder to detect, and it does not make real speech pauses shorter. A noisier corpus would not push these numbers down.

Silero's segmentation decides what counts as a pause, so a different segmenter would shift the individual gaps while leaving the shape intact — pauses of 1.2 s in read English speech (p95) are not an artefact of a 100 ms floor.

n is small: 124 pauses across 200 clips, all single utterances. Enough to refute 300 ms decisively, not enough to fix the optimal value precisely. **The actionable next question is not "can we lower it" but "should we raise it",** which this sample already suggests.

## Reproduce

```bash
uv run python bench/vad_pause_stats.py \
    --set "Vie-Eng=$HOME/Downloads/2026-10-24/Vie-Eng dataset/Vie-Eng/wav" \
    --set "Eng-Vie=$HOME/Downloads/2026-10-24/Vie-Eng dataset/Eng-Vie/wav"
```

Results land in `bench-results/vad-pause-stats/`. Note that silero's `read_audio` now requires `torchcodec` under torchaudio 2.11, so the script reads with `soundfile` and resamples only when needed.
