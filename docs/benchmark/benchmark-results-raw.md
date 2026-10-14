# Benchmark Results — aggregated tables

_Generated 2026-10-07 from `docs/benchmark/raw-results/*.csv`_

## Run inventory

| Dataset | Manifest | Runs | Rows | Items | Candidates |
|---|---|---|---|---|---|
| FLEURS ASR | `eval_data/eval_manifest_v1.json` | 5 | 1620 | 540 | moonshine-tiny-en-hf-cpu, moonshine-tiny-vi-hf-cpu, whisper-small-faster-whisper-cpu, zipformer-en-sherpa-onnx-cpu, zipformer-vi-30m-sherpa-onnx-cpu |
| FLEURS MT | `eval_data/mt_vi_en_eval_manifest.json` | 4 | 778 | 359 | m2m100-vi-en-ct2-cpu, opus-mt-vi-en-ct2-cpu |
| VIVOS ASR | `eval_data/vivos_vi_eval_manifest.json` | 3 | 2700 | 900 | moonshine-tiny-vi-hf-cpu, whisper-small-faster-whisper-cpu, zipformer-vi-30m-sherpa-onnx-cpu |

Beam width is not recorded per row, so it is not listed above. Each candidate ran at its own sweep-optimal setting (Opus-MT 5, M2M-100 4), so this is a best-config-per-model comparison rather than one shared setting. The ordering is unaffected — M2M-100 leads despite the narrower beam — but the absolute BLEU values are not comparable to published BLEU; see the report appendix.

## 1. FLEURS ASR — mean WER by condition

### `moonshine-tiny-en-hf-cpu` · EN

mean WER 0.3286 | mean latency 1.41 s | RTF 0.13 | peak RAM 566 MB

| condition | clean | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| clean | 0.2728 | - | - | - | - |
| steady | - | 0.2648 | 0.2905 | 0.4123 | 0.6479 |
| impulsive | - | 0.2578 | 0.2608 | 0.2738 | 0.2768 |

### `moonshine-tiny-vi-hf-cpu` · VI

mean WER 0.3998 | mean latency 1.76 s | RTF 0.12 | peak RAM 528 MB

| condition | clean | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| clean | 0.2815 | - | - | - | - |
| steady | - | 0.3058 | 0.3499 | 0.5013 | 0.7701 |
| impulsive | - | 0.3194 | 0.3398 | 0.3469 | 0.3835 |

### `whisper-small-faster-whisper-cpu` · EN

mean WER 0.2189 | mean latency 3.22 s | RTF 0.32 | peak RAM 923 MB

| condition | clean | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| clean | 0.1598 | - | - | - | - |
| steady | - | 0.1877 | 0.2140 | 0.2576 | 0.4039 |
| impulsive | - | 0.1842 | 0.1819 | 0.1849 | 0.1957 |

### `whisper-small-faster-whisper-cpu` · VI

mean WER 0.3835 | mean latency 3.69 s | RTF 0.27 | peak RAM 923 MB

| condition | clean | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| clean | 0.3013 | - | - | - | - |
| steady | - | 0.3134 | 0.4328 | 0.4445 | 0.6909 |
| impulsive | - | 0.3042 | 0.3113 | 0.3100 | 0.3429 |

### `zipformer-en-sherpa-onnx-cpu` · EN

mean WER 0.3523 | mean latency 0.64 s | RTF 0.06 | peak RAM 677 MB

| condition | clean | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| clean | 0.2857 | - | - | - | - |
| steady | - | 0.2715 | 0.3144 | 0.4274 | 0.6720 |
| impulsive | - | 0.3004 | 0.3005 | 0.2973 | 0.3017 |

### `zipformer-vi-30m-sherpa-onnx-cpu` · VI

mean WER 0.1280 | mean latency 0.58 s | RTF 0.04 | peak RAM 674 MB

| condition | clean | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| clean | 0.0997 | - | - | - | - |
| steady | - | 0.1146 | 0.1253 | 0.1477 | 0.2088 |
| impulsive | - | 0.1069 | 0.1097 | 0.1184 | 0.1206 |

## 2. FLEURS MT (vi→en) — BLEU

| run | n | mean BLEU | median | min | max | mean latency s | peak RAM MB |
|---|---|---|---|---|---|---|---|
| fleurs-mt/full/m2m100-vi-en-ct2-cpu | 347 | 17.50 | 14.12 | 1.95 | 60.13 | 0.97 | 1102 |
| fleurs-mt/full/opus-mt-vi-en-ct2-cpu | 347 | 9.84 | 7.10 | 0.72 | 56.00 | 1.64 | 399 |
| fleurs-mt/gold-set/m2m100-vi-en-ct2-cpu | 42 | 18.99 | 15.42 | 3.18 | 58.59 | 1.02 | 1102 |
| fleurs-mt/gold-set/opus-mt-vi-en-ct2-cpu | 42 | 10.78 | 7.13 | 1.10 | 36.72 | 1.58 | 397 |

## 3. VIVOS ASR (vi) — mean WER by condition

### `moonshine-tiny-vi-hf-cpu`

mean WER 0.2591 | mean latency 0.48 s | RTF 0.14 | peak RAM 510 MB

| condition | clean | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| clean | 0.2009 | - | - | - | - |
| steady | - | 0.1731 | 0.1909 | 0.2547 | 0.4981 |
| impulsive | - | 0.2084 | 0.2170 | 0.2598 | 0.3287 |

### `whisper-small-faster-whisper-cpu`

mean WER 0.3641 | mean latency 3.03 s | RTF 0.92 | peak RAM 923 MB

| condition | clean | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| clean | 0.3463 | - | - | - | - |
| steady | - | 0.3308 | 0.3364 | 0.3939 | 0.5214 |
| impulsive | - | 0.3415 | 0.3409 | 0.3360 | 0.3302 |

### `zipformer-vi-30m-sherpa-onnx-cpu`

mean WER 0.0564 | mean latency 0.14 s | RTF 0.04 | peak RAM 383 MB

| condition | clean | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| clean | 0.0513 | - | - | - | - |
| steady | - | 0.0498 | 0.0491 | 0.0608 | 0.0924 |
| impulsive | - | 0.0550 | 0.0482 | 0.0504 | 0.0503 |
