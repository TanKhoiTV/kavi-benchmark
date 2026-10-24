# Phase 6 — Denoising Gate Results

> **Date:** 2026-07-20 **ASR candidate:** faster-whisper Small int8 (CPU) **Eval set:** Smoke test (2 utterances × 5 conditions)

## Binary Gate Decision

| Condition | Raw WER | Wiener WER | RNNoise WER |
| ----------- | --------- | ------------ | ------------- |
| Clean | 40.82% | 44.15% | 41.93% |
| SNR 0 / impulsive | 44.06% | **39.61%** | 56.09% |
| SNR 0 / steady | 70.48% | 73.72% | 86.91% |
| SNR 5 / impulsive | 39.71% | **35.27%** | 48.41% |
| SNR 5 / steady | 50.68% | 50.68% | 66.04% |

**Noisy-conditions weighted-average WER (SNR 5 + SNR 0, both noise types):**

| Denoiser | Avg WER |
| ---------- | --------- |
| Raw | 51.23% |
| **Wiener** | **49.82%** |
| RNNoise | 64.36% |

## Gate Verdict

**ADOPT Wiener denoiser** (`prop_decrease=0.5`)

- Wiener beats raw on impulsive noise at both SNR levels (−4.4pp at SNR 0, −4.4pp at SNR 5)
- Wiener matches raw on steady noise at SNR 5, slight degradation at SNR 0
- Wiener degrades clean slightly (+3.3pp) — acceptable trade-off for noisy robustness
- RNNoise performs worse than raw across all conditions — **REJECT**

## Caveats

- Smoke test only (2 utterances, limited noise types); full evaluation recommended before the on-device denoiser pick is closed (see the [open-parameters register](../decisions/README.md#open-parameters))
- `prop_decrease=0.5` is the default; tuning may improve clean-speech preservation
- Consider disabling denoiser for clean/High-SNR conditions (adaptive threshold)

## Next Steps

1. Run full lean eval slice with Wiener denoiser enabled
2. Tune `prop_decrease` (0.3–0.7 range) on larger sample
3. Consider adaptive gating: enable denoiser only when estimated SNR < threshold
