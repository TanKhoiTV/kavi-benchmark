# Which statistic should decide the direction? Five candidates, and a negative result

Date 2026-10-29. Follows [`lid-share-sweep.md`](lid-share-sweep.md), which measured that `argmax(mean per-token log-prob)` needs ~70% Vietnamese audio before it says Vietnamese, and [`lid-code-switching-survey.md`](lid-code-switching-survey.md), which predicted that per-model normalisation (the S-norm shape) is the standard cure.

**The prediction fails.** Normalisation helps, but it moves the operating point by a few points, not by the 25%-to-70% gap. The comparison is not miscalibrated; it is measuring the wrong thing. Section 4 explains what follows, and it is a change of target rather than another statistic.

Run: `bench/lid_statistic.py`, 100 organiser clip pairs, both shipping checkpoints, 12 conditions, 1200 clips, one decode pass recording raw per-token log-probs so every statistic is computed post hoc from the same arrays.

## 1. The clean anchor, first

| statistic | clean VI recall | clean EN recall |
| --- | --- | --- |
| mean (current) | 99% | 99% |
| median | 96% | 99% |
| top-k 30% | 95% | 100% |
| **z-normalised** | **100%** | 99% |
| t-normalised | 99% | 99% |

On monolingual audio every candidate is at or near the ceiling, and that is what the scored set is: **the organisers' 200 clips contain one English token in 100 Vietnamese sentences and no true switches**. So the choice of statistic is irrelevant to the 45% quality pillar.

## 2. The mixed curve

P(says VI), contiguous | embedded, n=100 per cell:

| VI share | mean | median | top-k | **znorm** | tnorm |
| --- | --- | --- | --- | --- | --- |
| 0% | 1% | 1% | 0% | 1% | 1% |
| 5% | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 | 0 \| 0 |
| 15% | 3 \| 3 | 2 \| 2 | 2 \| 1 | 3 \| 3 | 3 \| 3 |
| 25% | 3 \| 2 | 3 \| 2 | 2 \| 1 | 3 \| 3 | 3 \| 2 |
| 40% | 3 \| 6 | 2 \| 4 | 3 \| 4 | **7 \| 7** | 3 \| 6 |
| 70% | 66 \| 83 | 56 \| 81 | 55 \| 81 | **72 \| 87** | 66 \| 83 |
| 100% | 99% | 96% | 95% | 100% | 99% |

Three findings:

1. **At a 25% Vietnamese share every statistic says Vietnamese 2-3% of the time.** Normalisation does not touch that, because it is not a scale problem: the English checkpoint is *genuinely more confident* on Vietnamese-accented audio than on its own clean English, so no monotone transform of the two means separates them.
2. **Z-normalisation is the best statistic at every point above 40%** and the only one with a perfect clean anchor. It is worth keeping — it buys 4-7 points at 70% share and removes the threshold's dependence on each model's own scale — but it does not change the operating point.
3. **Top-k is actively harmful**: at a 25% share its margins fall inside the thin band 27% (contiguous) and 10% (embedded) of the time, versus 0-1% for the others. Selecting the most confident tokens makes the two models *more alike* — both always have confident tokens — so the discriminator leaks away. This is a clean refutation of the "trust the confident tokens" intuition.

## 3. The bug this run nearly published

The first version of `bench/lid_statistic.py` mapped a clip where either model emitted nothing to `-inf`, i.e. "Vietnamese loses". It reported **90% clean-VI recall**. The truth is 99%: the English transducer emits nothing on **9 of 100 clean Vietnamese clips**, and an empty hypothesis is *maximum evidence for Vietnamese*, not a failure.

The error was caught by checking the anchor against an independent full-corpus decode rather than trusting the run's own summary — and the rule now lives in one function (`decide`) precisely because having it in five statistics is how one of them gets it wrong. Recorded here rather than quietly fixed: a comparison whose calibration anchor is broken can produce a plausible-looking curve that means nothing, and this one did.

## 4. What follows: change the target, not the statistic

The measurement reframes the problem instead of solving it. Two realistic cases, neither of which is what the sweep grid mostly measures:

| real case | acoustic content | what actually helps |
| --- | --- | --- |
| **Vinglish word** (`ship`, `video`, `book phòng`), 0.2-0.6 s, embedded in a Vietnamese sentence | Vietnamese audio with a nativised loanword: VI share 80-95% | nothing detection-specific — this should be the *clean* path, and the open question is only whether loanword density makes the English model confident enough to misroute |
| **verbatim English phrase** (a quoted instruction, a product name, 1-3 s) | a genuinely monolingual English span under Vietnamese accent | **temporal separation**: decide per clause, not per turn — the pipeline already cuts clauses for MT and TTS, so each decision sees mostly one language |

The construction the sweep spends most of its range on — one contiguous foreign block occupying 25-70% of a single segment, spliced across two different speakers — is **neither** of those. It is why the curve looks catastrophic at 25-70% and why a synthetic share grid cannot by itself settle the user's hypothesis that a Vietnamese-majority utterance is Vietnamese.

So the next step is not a sixth statistic. It is to test the **per-clause** decision on the same audio, which is also the granularity the shipping pipeline already produces, and to measure loanword density as a separate axis. Until a real code-switched recording exists, both are upper bounds.

## Reproduce

```text
uv run --with sherpa-onnx==2.0.1 --with sherpa-onnx-core==2.0.1 \
    python -m bench.lid_statistic --pairs 100 --workers 6
```

Needs the two shipping checkpoints on the host — they live on the test device and are pulled with `adb pull` as described in `lid-code-switching.md`. Writes `bench-results/lid-statistic/report.json`, including the raw per-token arrays, so a new statistic can be evaluated without re-decoding.

Every number here is decision quality, never latency: turnaround is a device measurement on one clock, and the artifact records no timing at all.
