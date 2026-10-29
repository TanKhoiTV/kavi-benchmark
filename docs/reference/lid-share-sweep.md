# Vietnamese share vs the confidence comparison: the mechanism is biased, not just miscalibrated

Measured 2026-10-29 on the host, shipping checkpoints (`asr-vi-apache`, `asr-parakeet`), 40 clip pairs per condition, Wilson 95% intervals. Script: [`../../bench/lid_share_sweep.py`](../../bench/lid_share_sweep.py).

**All numbers here are decision quality** — which language the comparison picks. None is a latency claim: turnaround has to be measured on the phone, in the app, on one clock. The artifact deliberately records no timing.

The hypothesis under test, from the team's field experience with this pair: *code-switching is asymmetric — an English speaker rarely drops into Vietnamese, a Vietnamese speaker routinely embeds English — so if the Vietnamese share of an utterance is above roughly 25%, it is Vietnamese with minimal doubt.*

## The curve

| VI share (duration) | P(says VI), contiguous | P(says VI), embedded |
| --- | --- | --- |
| 5% | 0% [0, 9] | 0% [0, 9] |
| 15% | 5% [1, 17] | 2% [0, 13] |
| **25%** | **8% [3, 20]** | **2% [0, 13]** |
| 30% | 8% [3, 20] | 5% [1, 17] |
| 40% | 5% [1, 17] | 10% [4, 23] |
| 50% | 25% [14, 40] | 20% [10, 35] |
| 70% | 70% [55, 82] | **88% [74, 95]** |

At a 25% Vietnamese share the comparison says Vietnamese 2-8% of the time. It needs about **70%** before it is reliable, and the "embedded" construction — short English spans inside Vietnamese, which is the realistic shape — is if anything *worse* at low shares and better at high ones.

## Why: the two means are not on the same scale

| condition | VI model mean | EN model mean | margin |
| --- | --- | --- | --- |
| clean VI | -0.110 | -0.448 | +0.338 |
| clean EN | -0.522 | **-0.188** | -0.334 |
| 25% VI embedded | -0.513 | **-0.064** | -0.448 |
| 70% VI embedded | -0.230 | -0.520 | +0.290 |

The English checkpoint is **more confident on mixed audio than on its own clean English** (-0.064 against -0.188), while the Vietnamese checkpoint degrades roughly in proportion to how much English it is hearing. So the comparison is not measuring "which model fits this audio better"; it is measuring "which model is less damaged", and the English model barely registers damage from Vietnamese audio.

That is the "confidently wrong" failure the design anticipated, in its worst form: not a weak signal that a threshold can sharpen, but a signal that points the wrong way across most of the range.

## What this invalidates

The detection mechanism recorded in ADR-033 is `argmax(mean per-token log-probability)` across two heterogeneous models. As **specified**, that comparison is not valid on this pair: the per-token mean is a property of each model's own confidence scale, not a likelihood of the audio under that model, and the two scales are not comparable. Both detection tiers — early lock and endpoint arbitration — rest on the same arithmetic, so a tier-1 pass would not have implied a tier-2 pass either; both would have inherited this bias.

Three consequences for the build:

1. **Do not wire the raw comparison into the pipeline as "tier 1".** It would route most Vietnamese-with-English utterances to English, which is the exact scenario the product exists for.
2. **A bias term is not sufficient.** Adding δ to the Vietnamese side does help — at δ=0.25 the 70%-VI case goes 88% → 98% and the 50% case 20% → 55% — but at a 25% share it needs δ≈0.45 to reach 55%, and by then 18% of clean English is misrouted. The hypothesis's qualitative claim survives (Vietnamese *is* under-detected and a Vietnamese-favouring prior helps); its 25% threshold does not.
3. **The fix is a comparable statistic, not a threshold.** Candidates, cheapest first and all testable on the same recorded margins plus one clean re-run: per-model z-normalisation against each model's own clean-language distribution; top-k token confidence rather than the mean (the foreign region contributes many low-confidence tokens to one model and few to the other); and a sequence-level likelihood ratio with length normalisation. If none produces a monotone curve that reaches 95% at register length, the mechanism should be replaced rather than re-weighted — and the honest replacement is text plausibility over *both* transcripts, which this pair happens to make easy (Vietnamese output carries diacritics, English does not).

## What survives

- **Clean audio is fine**: 97.5% correct direction on VI, 100% on EN. The mechanism works where it was measured.
- **Duration dominance is tracked** above ~50% share, monotonically, in both constructions.
- **Empty hypotheses are evidence**: a model that emits no tokens has lost the comparison outright, and that rule costs nothing.

## Limits of this measurement

- The constructions are **synthetic splices**, so they overstate how cleanly a switch can be heard. They are not evidence about how often the register is code-switched — only about what the mechanism does when it is.
- The axis is **duration share**, while the statistic is **per token**; on clean clips the English model emits roughly 3-4× as many tokens as the Vietnamese one for similar audio, so a 25%-duration share is a much smaller share of tokens. That confound is itself one of the candidate fixes above (top-k), and it is why the curve should not be read as a per-word share.
- One checkpoint pair, one clip set, n=40 per condition. The direction of the bias is consistent across both constructions and both orders, which is why it is reported as a mechanism finding rather than a noisy estimate.

## Reproduce

```bash
adb pull /sdcard/Android/data/com.kavi.app/files/asr-vi-apache eval_data/asr/
adb pull /sdcard/Android/data/com.kavi.app/files/asr-parakeet  eval_data/asr/
uv run --with sherpa-onnx==2.0.1 --with sherpa-onnx-core==2.0.1 \
    python -m bench.lid_share_sweep --pairs 40 --workers 4
```

Runs on CPU on purpose: `ys_log_probs` is the int8 CPU signal the phone produces, and sherpa-onnx's provider list is cpu/cuda/coreml, so an OpenVINO or DirectML run on the Arc GPU would measure different numerics. Worker processes give the throughput; the numerics stay the shipping ones.
