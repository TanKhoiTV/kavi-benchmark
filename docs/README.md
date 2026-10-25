# Documentation — benchmark track

This repository is the **host-side track**: the benchmark harness, the eval sets and the published results. The public program docs (contest brief, specification, onboarding) live in `aivoice-2026`; the on-device track (app, ADR register, QAIRT conversion) lives in the private `kavi-android`.

## Current — read these first

- `benchmark/benchmark-results-fleurs-vivos.md` — the published v1 results: per-candidate tables with notes, regenerated from `benchmark/raw-results/*.csv`.
- `benchmark/report/report.pdf` — the Typst report (source under `benchmark/report/`, layout in `benchmark/report-layout.md`).
- `benchmark/register-missing-candidates.md` — what is registered but not yet run.
- `decisions/` — the public register index: ADR-032 (MT model choice, pending) and the open-parameter rows that belong to a public claim.

## Hot reference — still consulted

- `reference/benchmarking-plan.md` — pre-decision corpus catalog and licensing notes (§3, §4.4–§4.5). The architecture it predates is recorded in the private ADR register.
- `reference/denoising-gate-results.md` — the host-side Wiener gate. This is the measurement ADR-018 points at; it stays public because it was measured here.
- `reference/beam-sweep-host-aggregate.md` — the citable reduction of the host beam sweep behind open parameter #6 (MT beam width).
- `reference/vad-benchmark.md` — how the VAD noise benchmark was built and scored (constructed audio, MUSAN noise, RIRS impulse responses, energy VAD, collar, borderline rule, commands, known limits).
- `reference/pause-distribution.md` — measured intra-utterance pauses behind `vad_speech_timeout_ms`: 300 ms is unsafe, 500 ms already truncates 2-3% of Vietnamese utterances, and neither can simply be moved.
- `reference/vad-results.md` — what the VAD benchmark measured and what it did not. This is the measurement behind ADR-022's revision note (the record itself is private in `kavi-android`); raw table and command record under `bench/results/vad/`.

## Cold reference — historical

- `reference/benchmarking-todo.md` — the v0 harness execution checklist (implemented).
- `reference/README.md` — why each file in `reference/` is still here.

## Not here any more

Moved during the 2026-10 re-organisation (TanKhoiTV/kavi-benchmark#129):

| Was | Now | Why |
| --- | --- | --- |
| `docs/onboarding*.md`, `docs/reference/additional-reading.md`, `archive/` | `aivoice-2026` | program layer |
| `docs/specifications.md`, `docs/reference/contest-info.md` | `aivoice-2026/docs/` (canonical copies already existed) | program layer; the duplicates are gone |
| `docs/android-*-plan.md`, `docs/ndk-conversion-runbook.md`, `docs/rtranslator-test-protocol.md`, `docs/issue-116-quantization-ab-plan.md`, `docs/reference/phase-4-qnn-plan.md`, `docs/reference/rtranslator-comparison.md`, `docs/decisions/ADR-001…028` | `kavi-android` (private) | on-device mechanism |
| `bench/qnn/**`, `scripts/env.py`, `scripts/qairt-env.*`, `.kavi.env.example` | `kavi-android` (private) | conversion toolchain travels with its only caller |
