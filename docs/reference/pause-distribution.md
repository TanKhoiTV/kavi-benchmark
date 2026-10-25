# Intra-utterance pauses and `vad_speech_timeout_ms`

`vad_speech_timeout_ms` is the trailing silence a VAD must observe before it declares an utterance finished. It sits on the critical path: at 500 ms it spends a quarter of the 2.0 s turnaround before translation or synthesis begins. The question this page answers is whether it can be lowered to 300 ms without cutting sentences short.

**Verdict: it cannot.** A 300 ms timeout ends the utterance mid-sentence on 7 of 100 Vietnamese clips and ~38 of 100 English clips in the organizer sample corpus. The current 500 ms value is also not safe — it truncates 2-3 Vietnamese clips and ~19 English clips per 100 — but the timeout is boxed in from both sides, so the fix is not a different number.

## Method

- **Detector:** silero-vad (the library default JIT model, 16 kHz) over the organizer sample corpus in `~/Downloads/2026-10-24/Vie-Eng dataset/`: 100 clips per direction, 16 kHz mono, mean 3.34 s (Vietnamese) and 5.96 s (English).
- **A pause** is the gap between two consecutive speech segments. Leading silence before the first segment and trailing silence after the last are excluded: they are not pauses, and the timeout is supposed to fire on them.
- **Two deliberate choices to avoid flattering a short timeout:**
  - `speech_pad_ms=0`. The library default of 30 ms pads every segment at both ends and would shrink each measured pause by up to 60 ms.
  - `min_silence_duration_ms` is one frame, so short silences are counted instead of being merged into the surrounding speech before they can be seen.
- **Resolution is 32 ms** — one Silero frame at 16 kHz — so every measured pause is a multiple of 32 ms and a timeout must clear the observed maximum by at least one frame.
- **Threshold sensitivity** is reported rather than assumed: 0.7 is the conservative direction, since a stricter detector drops more marginal speech and therefore reports longer pauses.
- **Inter-word versus inter-sentence pauses:** all 100 Vietnamese clips are single sentences (0 of 100 transcripts contain more than one sentence-final mark), so every Vietnamese pause in this table is *inside* a sentence and a timeout reaching it loses content. The English clips have no transcripts available (the supplied spreadsheets cover only the Vietnamese IDs, 0 of 100 English IDs match), so the English figures bound endpointing events but cannot separate intra-sentence pauses from two sentences read together.

## Results

Vietnamese (Vie-Eng, 100 clips, mean 3.34 s, all single-sentence):

| threshold | pauses | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 0.5 | 37 | 192 ms | 416 ms | 672 ms | 1120 ms | **1120 ms** |
| 0.7 | 34 | 192 ms | 384 ms | 576 ms | 672 ms | **672 ms** |

| candidate timeout | pauses reaching it (0.5) | clips truncated (0.5) | clips truncated (0.7) |
|---|---|---|---|
| 200 ms | 17 | 17 | 16 |
| 250 ms | 12 | 12 | 12 |
| 300 ms | 7 | **7** | **6** |
| 350 ms | 5 | 5 | 4 |
| 400 ms | 4 | 4 | 3 |
| 450 ms | 3 | 3 | 2 |
| 500 ms | 3 | **3** | **2** |

English (Eng-Vie, 100 clips, mean 5.96 s):

| threshold | pauses | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 0.5 | 154 | 224 ms | 608 ms | 768 ms | 1696 ms | **1792 ms** |
| 0.7 | 169 | 224 ms | 608 ms | 768 ms | 1792 ms | **1920 ms** |

| candidate timeout | pauses reaching it (0.5) | clips endpointed early (0.5) | clips endpointed early (0.7) |
|---|---|---|---|
| 300 ms | 54 | **38** | **39** |
| 500 ms | 24 | **19** | **21** |

## Verdict

**300 ms is unsafe, in both directions.** It cuts 7% of Vietnamese utterances mid-sentence and endpoints ~38% of English clips early. The Vietnamese figure is the load-bearing one: those clips are single sentences, so a cut there loses speech outright.

**500 ms is also unsafe, and that is today's setting.** It cuts 3 Vietnamese clips (VE-8762 has a 672 ms intra-sentence pause, VE-9469 544 ms, VE-8940 1120 ms) and endpoints ~19 English clips per 100. The Vietnamese 1120 ms gap is not robust — it disappears at threshold 0.7, which marks it as marginal speech rather than a clear silence — but the 672 ms gap is present at both thresholds. So the shipping configuration already truncates roughly 2-3% of Vietnamese utterances, which is a correctness issue independent of latency.

**There is no safe value below 500 ms**, which answers the "largest safe value" question with nothing: every candidate from 200 to 450 ms truncates more clips than 500 ms does. And 500 ms does not clear the Vietnamese maximum either; clearing 672 ms robustly would need a timeout of at least 704 ms (672 plus one frame).

## Why the timeout cannot simply be raised

The gate now passes at 1849 ms median with the 500 ms timeout in place (MT/TTS overlap, kavi-android#55). Raising the timeout to ~700 ms would return the Vietnamese truncation rate to zero for this corpus but costs 200 ms, taking the median gate to ~2049 ms and out of budget. Lowering it to 300 ms returns 200 ms but breaks 7% of Vietnamese utterances.

**The timeout is boxed in from both sides.** The 200 ms the issue hoped to recover is not available for free, and the gate no longer needs it: the overlap work found the time elsewhere. The saving is therefore *stated but declined*: 300 ms would take the measured median gate from 1849 ms to 1649 ms, at the cost of the truncation above.

The remaining option is not a number but a rule: endpoint on evidence that the sentence is finished — a non-growing ASR hypothesis, or a decoder end-of-sentence token — rather than on silence alone. That is filed separately; this page establishes only that the arithmetic ("300 ms is free below the observed p99") is wrong.

## Limitations

- **Clean read speech.** The corpus has no factory or construction noise. Under noise the detector drops marginal speech, which lengthens pauses, so these numbers are a lower bound on pause duration in the target conditions. The on-device noisy measurement is the remaining evidence gap (kavi-android#47).
- **Host Silero, not the on-device build.** Frame size and model differ from the sherpa-onnx build shipped in the app, so device pause detection will not reproduce these exact values.
- **Small tail.** 100 clips per direction means p99 is a single clip; only the maximum and the per-clip counts should be quoted with confidence.
- **English cannot separate intra-sentence from inter-sentence pauses** for want of transcripts, so the English truncation counts are an upper bound.

## Reproduction

```bash
python -m bench.pause_distribution \
    --wav-dir "VI=.../Vie-Eng dataset/Vie-Eng/wav" \
    --wav-dir "EN=.../Vie-Eng dataset/Eng-Vie/wav" \
    --out bench-results/pause-distribution.json
```

`--thresholds` defaults to `0.5 0.7`; `--limit` shortens the run. The JSON carries every per-clip pause list, the per-candidate counts, and the per-corpus summary used above.
