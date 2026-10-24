# VAD benchmark results (energy VAD, ADR-022)

How the benchmark is built and scored is in [vad-benchmark.md](vad-benchmark.md).
This page records what was measured and what can and cannot be concluded.

## Status

| Item | Status |
|---|---|
| Stage 3A: manifest of 800 items, tests, reproducibility, coverage | Done. `make check` and `make test` green (177 tests at the sweep commit), manifest SHA256 identical across two independent generations, `make vad-audit` reports 0 problems |
| Stage 3B: collar measured from 50 hand-labelled gold clips | **Not performed.** 0 of 50 clips labelled. Every result below uses the provisional collar of 0.150 s |
| Stage 4: energy VAD, VAD-only runner, smoke test, 5 x 5 sweep | Done and run on the real data |
| Pause rule, sensitivity of results to the collar | Measured (below) |
| Comparison with a second VAD (for example Silero) | Not done |
| End-to-end turnaround under 2000 ms | Not measured. It needs ASR, MT and TTS; this runner stops at the VAD |

## What was run

- Commit `9a6278d` (sweep) plus the collar sensitivity analysis.
- Manifest `eval_data/vad_manifest_v2.json`, 800 items, SHA256
  `706099670f5558a1ef2ecf435b2da1984d7fefa57b740e92e6537608ed737669`
  (`eval_data` is not committed; it is regenerated with `make vad-manifest`).
- Ground truth: speech bounds from silero-vad 6.2.1 with library defaults, stored in
  `bench/vad_source_bounds.json` for 60 FLEURS clips. Only 20 clips (10 VI, 10 EN)
  are used, and the same 20 appear in every cell.
- Level: each clip scaled to p99 frame peak 0.35. After RIR the level is higher
  (0.54 to 0.66 in the 10 representative items).
- Representative audio validation: 10 of 10 pass, SNR error at most 0.00004 dB.
- Sweep: energy_threshold in 0.01, 0.02, 0.05, 0.10, 0.20 and speech_timeout_ms in
  200, 350, 500, 750, 1000, 800 items each. ADR default is (0.05, 500). Wall time
  55.5 s.
- Criteria used: missed onset <= 5%, false triggers <= 0.1 per minute, clipped speech
  <= 50 ms per utterance (mean over items). The Stage 1 text was not available, so
  these definitions and the borderline margins in `bench/vad_borderline.py` are
  assumptions.

## Results at the ADR default (0.05, 500 ms), collar 0.150 s provisional

| Condition / SNR | False triggers per min | Missed onset % | Clipping ms | Pause not bridged (pause <= timeout) | No end of speech (of 80) |
|---|---|---|---|---|---|
| quiet / clean | 0.00 | 0.0 | 18 | 31 | 0 |
| quiet / +15 | 6.38 | 10.0 | 15 | 24 | 15 |
| street / +10 | 27.00 | 65.0 | 5065 | 10 | 49 |
| street / +5 | 38.25 | 100.0 | 4146 | 5 | 73 |
| indoors / +10 | 21.00 | 70.0 | 0 | 0 | 80 |
| indoors / +5 | 21.00 | 70.0 | 0 | 0 | 80 |
| near-field / +5 | 21.00 | 70.0 | 0 | 0 | 80 |
| near-field / 0 | 21.00 | 70.0 | 0 | 0 | 80 |
| far-field / 0 | 37.12 | 100.0 | 1111 | 1 | 78 |
| far-field / -5 | 37.50 | 100.0 | 810 | 5 | 78 |

Reading notes:

- Clipping of 0 in indoors and near-field is not a good result. The VAD never closes
  there (no end of speech in 80 of 80 items), so no end can be clipped.
- indoors/+10, indoors/+5, near-field/+5 and near-field/0 give identical numbers at
  this configuration. These cells do not separate the SNR levels; the cause
  (suspected: saturation by the steady noise files assigned to them) was not
  investigated.

## Best grid point per condition (lowest missed onset, then false triggers)

| Condition / SNR | Threshold, timeout | False triggers per min | Missed onset % | Clipping ms | Configurations (of 25) meeting all three criteria |
|---|---|---|---|---|---|
| quiet / clean | 0.01, 200 | 0.00 | 0.0 | 0 | 15 |
| quiet / +15 | 0.05, 200 | 12.38 | 0.0 | 15 | 0 |
| street / +10 | 0.10, 200 | 21.00 | 0.0 | 52 | 0 |
| street / +5 | 0.20, 200 | 24.75 | 10.0 | 244 | 0 |
| indoors / +10 | 0.20, 1000 | 3.00 | 20.0 | 65 | 0 |
| indoors / +5 | 0.20, 200 | 27.00 | 31.2 | 184 | 0 |
| near-field / +5 | 0.20, 200 | 27.00 | 31.2 | 184 | 0 |
| near-field / 0 | 0.10, 200 | 25.50 | 65.0 | 922 | 0 |
| far-field / 0 | 0.20, 200 | 47.62 | 65.0 | 1369 | 0 |
| far-field / -5 | 0.20, 200 | 48.75 | 86.2 | 2487 | 0 |

## Whole benchmark, 800 items (collar 0.150 s provisional)

These averages are dominated by the nine noisy condition groups.

Missed onset %, rows are thresholds, columns are timeouts in ms:

| | 200 | 350 | 500 | 750 | 1000 |
|---|---|---|---|---|---|
| 0.01 | 83.2 | 84.8 | 84.8 | 84.8 | 85.0 |
| 0.02 | 71.0 | 75.0 | 76.0 | 76.0 | 76.0 |
| 0.05 | 57.1 | 62.0 | 65.5 | 66.5 | 67.0 |
| 0.10 | 41.5 | 49.6 | 52.5 | 56.6 | 57.1 |
| 0.20 | 33.9 | 39.8 | 42.2 | 46.2 | 47.4 |

False triggers per minute:

| | 200 | 350 | 500 | 750 | 1000 |
|---|---|---|---|---|---|
| 0.01 | 31.65 | 27.82 | 25.91 | 25.50 | 25.50 |
| 0.02 | 28.12 | 26.70 | 24.26 | 22.80 | 22.80 |
| 0.05 | 26.77 | 24.41 | 23.02 | 20.74 | 20.14 |
| 0.10 | 24.97 | 21.94 | 20.40 | 17.85 | 16.84 |
| 0.20 | 22.72 | 18.26 | 16.39 | 13.35 | 12.07 |

No grid point meets the false-trigger criterion (0.1 per minute) or the missed-onset
criterion (5%) over the whole benchmark.

## Conclusions that do not depend on the collar

- Every condition other than quiet/clean fails the three criteria at every one of the
  25 grid points, and at every collar from 0 to 200 ms (0 of 25 configurations meet
  them in each of those nine groups, at every collar tested).
- Where the VAD fails, it stays open: end of speech is never declared in 49 to 80 of
  80 items at the ADR default in street, indoors, near-field and far-field.
- The rule "a pause no longer than the timeout must not split" is violated whenever
  the timeout exceeds the pause by less than about 300 ms. The share of items whose
  A-B pause was not bridged depends only on timeout minus pause:

  | timeout - pause | threshold 0.01 | threshold 0.05 |
  |---|---|---|
  | 0 ms | 60% | 95% |
  | 50 ms | 30% | 85% |
  | 150 ms | 5% | 60% |
  | 250 ms | 0% | 5% |
  | 300 ms or more | 0% | 0% |

  Each cell has 20 items, so every percentage is a multiple of 5. Timeout 1000 ms
  shows no violation, but end of speech is then undeclared in 8 to 48 of 80 clean
  items because the benchmark's trailing silence is only 1000 ms.
- At the ADR default in quiet/clean, 31 of the 60 items with pause <= 500 ms were not
  bridged (about 52%).

## The conclusion that depends on the collar

Quiet/clean at the ADR default (0.05, 500 ms), by collar:

| Collar ms | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|
| Missed onset % | 80 | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| Clipping, mean ms | 156 | 118 | 92 | 72 | 53 | 34 | 18 | 4 |

The ADR default meets all three criteria in quiet/clean only for a collar of about
105 ms or more (clipping 53 ms at 100 ms is borderline; 34 ms at 125 ms passes). The
smallest collar in the grid from which it passes and keeps passing is 125 ms. The best
of the 25 configurations passes from 50 ms. **The true collar was not measured**, so
whether the ADR default passes in quiet/clean is **undetermined**. If the reference
boundaries are within a few tens of milliseconds of what a listener marks, the
collar is below 105 ms and the ADR default fails the clipping criterion even in clean
audio. That is an expectation, not a result.

## Limitations

- Stage 3B was not performed, so the collar, and with it every missed-onset and
  clipping figure, is provisional.
- The reference boundaries come from another VAD (Silero with 30 ms padding), not from
  listeners. They are a proxy for speech boundaries.
- Only 20 source clips; false-trigger figures are per 2 s of non-speech per item. A
  claim of 0.1 false triggers per minute or less at 95% confidence would need about 30
  minutes of non-speech per condition group, against about 2.7 minutes available.
- The level, SNR and borderline definitions are assumptions listed in
  [vad-benchmark.md](vad-benchmark.md) and need checking against the Stage 1 documents.
- The failure of the noisy conditions is established for this benchmark and this energy
  VAD. It does not show the benchmark is fair, or that no VAD design passes. That needs
  the same manifest scored with a second VAD.
- ADR-022 fixes only an amplitude threshold and a silence period; frame size, energy
  definition, hang-in and hang-out are assumptions in `bench/vad_energy.py`.

## Reproduce

```bash
make vad-bounds
make vad-manifest
make vad-audit
make vad-sweep
make vad-collar-sensitivity
```

`make vad-bounds` needs torch and silero-vad; the committed bounds file makes it
optional. Output goes to `bench-results/`.
