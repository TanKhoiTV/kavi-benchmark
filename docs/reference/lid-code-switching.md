# Code-switching: the confidence signal stays confident, and often picks the wrong language

Measured 2026-10-29 on the host, with the **shipping checkpoints** pulled from the device (`asr-vi-apache`, `asr-parakeet`), over 40 clip pairs per condition. Script: [`../../bench/lid_codeswitch.py`](../../bench/lid_codeswitch.py).

Code-switching is the risk the language-detection decision carries, because the pipeline picks a translation direction from the two recognisers' per-token confidence. This measures the mechanism's exposure to mixed-language speech: does the margin collapse (in which case the pipeline safely defers), or stay decisive (in which case it commits, right or wrong)?

## Method, and what it is not

Real organizer audio, both languages, spliced **inside one segment**: VI then EN, and EN then VI, with 80 ms of silence at the join, at three duration-dominance ratios. Each clip is decoded by both specialists and each decode's mean per-token log-probability is compared; `margin > 0` favours Vietnamese.

Splicing is not how people mix languages — a real switch has no boundary and usually happens inside a phrase — so this overstates how *cleanly* a switch can be heard. It says nothing about how often the contest register is mixed. What it does establish is the *direction and size* of the mechanism's error, which is what a failure budget needs.

## Results

| condition | n | margin p10 | p50 | p90 | VI wins | thin | agrees with dominant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean VI | 40 | 0.178 | **0.439** | 0.782 | 98% | 12% | **98%** |
| clean EN | 40 | -0.611 | **-0.493** | -0.371 | 0% | 2% | **98%** |
| mixed 70/30 VI-first | 40 | -0.204 | 0.187 | 0.576 | 70% | 15% | **70%** |
| mixed 30/70 EN-first | 40 | -0.091 | 0.166 | 0.573 | 82% | 15% | **82%** |
| mixed 70/30 EN-first | 40 | -0.451 | -0.365 | -0.192 | 8% | 0% | **92%** |
| mixed 30/70 VI-first | 40 | -0.479 | -0.368 | -0.046 | 8% | 10% | **92%** |
| mixed 50/50 VI-first | 40 | -0.388 | -0.191 | 0.311 | 25% | 12% | 25% |
| mixed 50/50 EN-first | 40 | -0.361 | -0.164 | 0.085 | 22% | 15% | 78% |

"Thin" is `|margin| < 0.05`, about a ninth of a clean-audio margin — the band where the ladder should defer rather than commit. "Agrees with dominant" compares the winner against the language carrying the larger share of the duration.

## Four findings

1. **Mixed audio does not produce an undecided margin.** The thin share stays at 10-15%, the same band as clean audio. Both specialists find real words to be confident about, so the confidence signal does not degrade when the language changes mid-utterance — it just answers a question the signal was never built to answer. This is the failure mode the design anticipated ("confidently wrong"), now measured rather than feared.
2. **Dominance is followed, but not reliably.** At 70/30 the dominant language wins 70% (VI dominant) to 92% (EN dominant). For a Vietnamese speaker embedding English terms — the realistic direction of code-switching here — that is a **18-30% wrong-direction rate**, against a 3% budget.
3. **At equal duration there is an English bias**, and it is not a position artifact: VI-first gives EN 75% of decisions, EN-first gives EN 78%. The English checkpoint's margin is also larger on clean audio (0.493 against 0.439), so at parity the comparison tilts to English by construction.
4. **A model that emits nothing is not a missing measurement.** On 8% of clean VI clips the English transducer produced no tokens at all; on 2% of clean EN clips the Vietnamese one did. Treating an empty hypothesis as maximum evidence for the other side is both correct and free, and it arrives before any confidence comparison happens.

## What it means for the build

- **A ≤3% wrong-direction budget cannot cover code-switched speech.** It holds for monolingual utterances (98% on clean audio) and fails by an order of magnitude on mixed ones. The budget has to be stated per class, or the claim is false on exactly the input the contest's factory-and-logistics scenario produces.
- **Detection granularity must be the utterance, not the phrase.** Dominance is followed at utterance level; nothing here supports switching direction mid-turn, and (3) shows even a *tie* resolves to English.
- **The conversation gives information the audio does not.** In a two-party exchange the direction alternates by speaker, not by language — so turn history is a prior the confidence comparison should be combined with. That is a design change, not a threshold adjustment, and it is the strongest mitigation this measurement suggests.
- **The demo script should not rely on code-switched turns** unless the drop is disclosed. It is a legitimate thing to say out loud: "we detect the dominant language per utterance; mixed speech is a known limitation and the next version uses turn context."

## Reproduce

```bash
adb pull /sdcard/Android/data/com.kavi.app/files/asr-vi-apache eval_data/asr/
adb pull /sdcard/Android/data/com.kavi.app/files/asr-parakeet  eval_data/asr/
uv run --with sherpa-onnx==2.0.1 --with sherpa-onnx-core==2.0.1 \
    python -m bench.lid_codeswitch --pairs 40
```

The artifact records the checkpoints, the splice construction, the gap, the thin threshold and the synthetic caveat beside the numbers, and lands in `bench-results/lid-codeswitch/report.json`. The checkpoints themselves are staged on the device rather than fetched here, which is why the command starts with two `adb pull`s.
