# MT encoder on HTP v73 — Qualcomm AI Hub profile

**Date:** 2026-10-24 · **Device:** Samsung Galaxy S23 Family (`chipset:sm8550`, `hexagon:v73`, `soc-model:43`) · **Model:** `opus-mt-vi-en` encoder · **Precision:** `w8a16` (20 calibration samples) · **Inputs:** `input_ids` + `attention_mask`, fixed `(1, 128)` · **Compile job:** `j5q1edjeg` · **Profile job:** `jp11ymrnp`

| Metric | Value |
| --- | --- |
| Inference p50 (99 steady runs after 1 warmup) | **2.03 ms** |
| Inference p95 | 2.06 ms |
| Inference range | 2.02 – 2.24 ms (11% of median) |
| First inference (warmup) | 3.60 ms |
| Compute unit | **NPU — 224 of 224 ops, zero CPU fallback** |
| Peak memory (inference) | 174.9 MB |
| Context-binary load, first | **447 ms** |
| Context-binary load, warm | **390 ms** |
| Context-binary size | 47.2 MB |

The device reports the same SoC as the project's target Meizu 21 Note, so this is same-silicon evidence rather than a proxy for a different chip.

## Verdict

The encoder costs **2.03 ms** on the NPU, against a measured **507 ms** decoder on CPU for the same model. The encoder is **0.4% of the MT stage**; the decoder is the other 99.6%.

| Stage | Measured | Share of MT |
| --- | --- | --- |
| Opus-MT encoder (HTP v73) | 2.03 ms | 0.4% |
| Opus-MT decoder (CPU, beam=1, guarded) | 507 ms | 99.6% |
| M2M-100 encoder (HTP, projected) | ~8 ms | 0.5% |
| M2M-100 decoder (CPU, beam=1, guarded) | 1 562 ms | 99.5% |

This confirms ADR-023's split (encoder on HTP, decoder on CPU) as correct, and simultaneously shows the split buys almost nothing on latency: the stage it accelerates was never the bottleneck. It also closes one line of reasoning — **M2M-100 cannot be rescued by moving its encoder to the NPU**, because its 1 562 ms is decoder time.

The M2M-100 encoder row is a **projection** from Opus-MT geometry (12 layers / 1024 dim against 6 / 512), not a measurement. Measuring it only matters if M2M-100 is reconsidered for VI-ZH/VI-KO coverage.

## Two corrections to the memory and startup budget

`kavi-tuning.yaml` carries `mt_encoder_mb: 46`; the context binary measures **47.2 MB**, so the weight-footprint prediction is excellent. But inference peak memory is **174.9 MB**, 3.7× that line item — the difference being HTP runtime, compiled-graph workspace, and mapped libraries inside the measured process. That line item is a weight footprint, not a residency cost, and ADR-016's per-component table understates the encoder's real contribution.

**Context-binary load is 390–447 ms.** That is two-thirds of the 500 ms VAD timeout and a fifth of the 2.0 s turnaround, spent before the first utterance can be translated. Acceptable as a one-time cost at app start; it must never be a lazy load on first use.

## Why the NPU figure is citable

Every one of the 224 profiled ops reports `compute_unit: NPU`. Had any op fallen back to CPU, the 2.03 ms would describe a partition of the graph rather than the hardware, which is the failure mode that makes accelerator numbers unreliable in general.

## What this still cannot claim

**Not a turnaround figure.** This profile covers the encoder alone. It excludes capture, VAD, ASR, the CPU decoder, TTS, and every handoff. The MT stage is the encoder plus a 507 ms decoder, and the 2.0 s gate covers the whole pipeline.

**Not the target device.** Same SoC, different phone: thermal headroom and clock policy differ, so treat this as same-silicon evidence and re-measure on the Meizu 21 Note through the app (kavi-android#40) if a device-specific number is needed.

**Per-op times are relative weights, not additive truth.** `execution_detail` attributes the largest share to attention softmax (~585 µs × 6 layers) and only 132 µs to the largest MatMul, but the op times sum to 6.98 ms against 2.03 ms of wall clock. The useful signal is that the encoder is elementwise/memory-bound on softmax rather than GEMM-bound — which is exactly why it is so cheap to keep on the NPU.

## Reproduce

```bash
uv run --group qairt qai-hub configure --api_token "$QAI_HUB_API_TOKEN"
uv run --group qairt python bench/aihub_profile.py \
    --onnx ../kavi-android/models/qnn/opus-mt-vi-en/encoder_model.onnx \
    --calib <dir with ids_<n>.raw / mask_<n>.raw>
```

Re-submission is cheap after the first run: `--model-id`, `--dataset-id`, and `--compile-job-id` skip the 186 MB upload, the dataset upload, and the compile respectively. Results land in `bench-results/aihub-encoder/{profile.json,summary.json}`; `--fp16` compiles without calibration for a precision comparison.

Conversion contract worth knowing, each learned by rejection: `--target_runtime precompiled_qnn_onnx` (not `qnn_context_binary`), `--quantize_full_type w8a16` (calibration data is rejected without it), `--truncate_64bit_io` (the graph declares int64 inputs, HTP has no 64-bit integers), `input_specs` carry dtype as `((shape), "dtype")`, and calibration entries must match that spec dtype with a leading batch dimension.
