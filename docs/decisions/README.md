# Architecture Decision Records

One decision per record. This index is the entry point: it lists every ADR with
its current status, shows the supersession graph, and records the open parameters
that no ADR has closed yet.

Records live alongside this file as `ADR-NNN-<slug>.md`. Decisions are never
deleted and numbers are never reused — a decision that is replaced points at its
successor, and a record that was split points at the records it became.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| **Proposed** | Written and under consideration; not yet binding |
| **Accepted** | Binding. May carry an inline revision note if later amended |
| **Superseded by ADR-NNN** | Replaced. Retained as the historical record |
| **Withdrawn** | Not a decision after all; retained for provenance |

An ADR may be *Accepted* and still have an **open parameter** — a sub-choice
(model binary, threshold value) that was deliberately left to measurement. Those
are listed in [Open parameters](#open-parameters) rather than being invented as
decisions.

## Index

| ADR | Decision | Status |
| --- | --- | --- |
| [ADR-001](ADR-001-offline-first-on-device-architecture.md) | Offline-first, on-device architecture | Accepted |
| [ADR-002](ADR-002-target-platform.md) | Target platform — Snapdragon 8 Gen 2 / Android 16 / Hexagon NPU | Accepted |
| [ADR-003](ADR-003-hexagon-runtime.md) | Hexagon runtime / compiler strategy | Proposed |
| [ADR-004](ADR-004-architecture.md) | *(withdrawn — was the deferred tech-stack register)* | Withdrawn |
| [ADR-005](ADR-005-qnn-isnan-workaround.md) | Strip the unsupported `IsNaN` op when converting to QNN | Accepted |
| [ADR-006](ADR-006-native-on-device-runner.md) | Native on-device runner (no ADB bridge) | Accepted |
| [ADR-007](ADR-007-translation-service.md) | TranslationService — two-mode foreground service | Accepted |
| [ADR-008](ADR-008-dual-zipformer-asr.md) | v1 Android ASR — dual Zipformer | Accepted |
| [ADR-009](ADR-009-supertonic-tts-v1.md) | v1 Android TTS — SupertonicTTS 3 (Phase 1) | Proposed |
| [ADR-010](ADR-010-all-opt-decoder-optimisation.md) | `ALL_OPT` is safe — no custom ONNX operators | Proposed |
| [ADR-011](ADR-011-android-asset-provenance-delivery.md) | Android model asset provenance & delivery | Accepted |
| [ADR-012](ADR-012-asr-thread-tuning.md) | ASR thread tuning strategy for SD8G2 | Proposed |
| [ADR-013](ADR-013-persistent-model-residency.md) | Persistent model residency — load at startup | Accepted |
| [ADR-014](ADR-014-max-size-kv-cache-preallocation.md) | Max-size KV cache pre-allocation | Accepted *(modified by ADR-008)* |
| [ADR-015](ADR-015-npu-cpu-zero-copy-ion.md) | NPU→CPU zero-copy handoff via ION | Accepted |
| [ADR-016](ADR-016-memory-budget.md) | Peak memory budget | Accepted *(revised by ADR-008)* |
| [ADR-017](ADR-017-dual-asr-language-detection.md) | Dual ASR inference for language detection | **Superseded by ADR-008** |
| [ADR-018](ADR-018-denoising-slot.md) | Tiered, toggleable denoising slot | Accepted *(model open)* |
| [ADR-019](ADR-019-qnn-runtime-bundling.md) | QNN runtime bundling — jniLibs roster + version lock | Accepted *(ASR roster withdrawn by ADR-027)* |
| [ADR-020](ADR-020-pipeline-concurrency.md) | Pipeline concurrency + per-encoder CPU fallback | Accepted |
| [ADR-021](ADR-021-ort-cpu-decoder-session-options.md) | ONNX Runtime CPU decoder session options | Accepted |
| [ADR-022](ADR-022-energy-vad.md) | Energy-based VAD with speech timeout | Accepted *(no energy threshold fixed — model-based VAD follow-up)* |
| [ADR-023](ADR-023-decoder-on-cpu.md) | Encoder on NPU, decoder on CPU | Accepted *(ASR half superseded by ADR-027)* |
| [ADR-024](ADR-024-piper-tts-on-cpu.md) | Piper TTS stays on CPU | Accepted |
| [ADR-025](ADR-025-app-layer-dynamic-padding.md) | App-layer dynamic padding/trimming for fixed shapes | Accepted |
| [ADR-026](ADR-026-real-data-calibration.md) | Calibration must use real data only | Accepted |
| [ADR-027](ADR-027-asr-cpu-only.md) | ASR stays CPU-only for v1 | Accepted |
| [ADR-028](ADR-028-vieneu-tts-migration.md) | Migrate v1 TTS to VieNeu-TTS v3 Turbo (Phase 2) | Proposed |
| [ADR-029](ADR-029-license-gate.md) | Shipping license gate — adopt/avoid verdicts | Accepted |
| [ADR-030](ADR-030-qairt-runtime-redistribution.md) | QAIRT runtime redistribution is permitted | Accepted |
| [ADR-031](ADR-031-piper-engine-gpl-split.md) | Piper engine GPL split | Proposed *(deferred)* |

**Not an ADR:** [`docs/reference/rtranslator-comparison.md`](../reference/rtranslator-comparison.md)
— an architectural comparison with RTranslator 2.1.5. It contains analysis and
adopted patterns, not a decision, and previously collided with the ADR-007 number.

## Supersession graph

```text
ADR-007 (TranslationService) ──split──▶ ADR-013 … ADR-022
     │
     └─ Decision 6 ─▶ ADR-017 ──superseded by──▶ ADR-008
                                                  │
                                   ┌──────────────┼───────────────┐
                                   ▼              ▼               ▼
                            ADR-014 (rev.)  ADR-016 (rev.)   ADR-027 (ASR CPU-only)
                                                                  │
                                                      revises ADR-019 + ADR-023

ADR-009 (TTS Phase 1) ──split──▶ ADR-028 (TTS Phase 2 migration)
ADR-010 expands ADR-021

license-situation ──split──▶ ADR-029 (gate) · ADR-030 (QAIRT) · ADR-031 (Piper fork)

ADR-004 (deferred tech-stack) ──withdrawn──▶ superseded by ADR-008 · ADR-009 · ADR-030
```

## Split provenance (2026-09-25)

Several records held multiple decisions, which made their file-level status
misleading — most notably ADR-007, which was labelled *"Superseded by ADR-008"*
even though ADR-008 supersedes only one of its eleven decisions. Each was split
into one decision per record; the originals were renamed (retaining their number)
and carry a split note.

| Original | Became |
| --- | --- |
| ADR-005 `Decision 1–5` | [ADR-005](ADR-005-qnn-isnan-workaround.md) + [ADR-023](ADR-023-decoder-on-cpu.md) + [ADR-024](ADR-024-piper-tts-on-cpu.md) (the epsilon guard became a consequence; the padding decision deduplicated into ADR-025) |
| ADR-006 `Decision 1–4` | [ADR-006](ADR-006-native-on-device-runner.md) + [ADR-025](ADR-025-app-layer-dynamic-padding.md) + [ADR-026](ADR-026-real-data-calibration.md) (the epsilon guard deduplicated into ADR-024) |
| ADR-007 `Decision 1–11` | [ADR-007](ADR-007-translation-service.md) + [ADR-013](ADR-013-persistent-model-residency.md) … [ADR-022](ADR-022-energy-vad.md) |
| ADR-008 (+ amendments) | [ADR-008](ADR-008-dual-zipformer-asr.md) + [ADR-027](ADR-027-asr-cpu-only.md); the ADR-007 amendments became revision notes in ADR-014/ADR-016 |
| ADR-009 Phase 1/2 | [ADR-009](ADR-009-supertonic-tts-v1.md) + [ADR-028](ADR-028-vieneu-tts-migration.md) |
| `license-situation.md` | [ADR-029](ADR-029-license-gate.md) + [ADR-030](ADR-030-qairt-runtime-redistribution.md) + [ADR-031](ADR-031-piper-engine-gpl-split.md) |
| `ADR-007-vs-RTranslator-comparison.md` | moved to [`docs/reference/rtranslator-comparison.md`](../reference/rtranslator-comparison.md) |

## Open parameters

Decisions deliberately left unclosed. Each is owned by an ADR; they are collected
here so nothing is silently forgotten. This register replaces the former
`ADR-004`, which was a parameter table rather than a decision.

| # | Open parameter | Owned by | Closes when |
| --- | --- | --- | --- |
| 1 | **Final MT model** | *no record yet* | An MT decision ADR is written (tracked as issue #110). Opus-MT is in use and pinned in `.kavi.yaml`, but the choice was never recorded — ADR-004 deferred it and ADR-008/ADR-009 closed only ASR and TTS. *Not measurement-gated: it needs a record, not a number.* |
| 2 | **On-device denoiser model** (GTCRN vs Wiener) | [ADR-018](ADR-018-denoising-slot.md) | **Needs a number:** on-device WER for GTCRN vs Wiener on the same clips. The host-side Wiener gate is already recorded in [`docs/reference/denoising-gate-results.md`](../reference/denoising-gate-results.md). *(issue #111)* |
| 3 | **Piper engine licence path** | [ADR-031](ADR-031-piper-engine-gpl-split.md) | **Downstream of the TTS decision — not currently blocking.** Piper is fallback-only in the working position, so this closes when the TTS choice is ratified: dropped from the shipped stack → closes outright; kept as fallback → the MIT-era `rhasspy/piper` snapshot must be pinned and subprocess isolation adopted. Urgent only if Supertonic misses the RTF ≤ 0.5 gate in `.kavi.yaml`. *(issue #112)* |
| 5 | **ASR thread-tuning build** (B vs F) | [ADR-012](ADR-012-asr-thread-tuning.md) | **Needs a number:** on-device A/B on the Meizu 21 Note, 100 utterances per mode, logging RTF, thermal status and audio-overrun counters. Build B is already the working config, so this validates it and decides whether the Build F escalation is needed; the ADR's nine platform questions also need confirming on-device. *(issue #114)* |
| 6 | **MT beam width** (greedy vs beam=4) | [ADR-020](ADR-020-pipeline-concurrency.md) | **Needs a number:** Phase-5 latency-vs-BLEU measurement. *(issue #115)* |
| 7 | **w8a16 vs w8a8** for Opus-MT | [ADR-003](ADR-003-hexagon-runtime.md) | **Needs a number:** accuracy regression measurement. *(issue #116)* |

> **Resolved — removed from the table.** **#4 VAD thresholds:** closed by the
> host-side noise benchmark *(issue #113)*. No energy threshold and speech timeout
> meets the criteria in the noisy conditions, so none is fixed and a model-based VAD
> is the follow-up (not yet tracked). The collar was not measured and the evidence is
> host-side only, so on-device confirmation is pending — see the revision note in
> [ADR-022](ADR-022-energy-vad.md).
>
> **Resolved — removed from the table.** **#8 QAIRT Community Edition access:**
> obtained. The SDK (`2.31.0.250130`) is installed and in use — see
> [`docs/ndk-conversion-runbook.md`](../ndk-conversion-runbook.md) §2.2 and
> `scripts/qairt-env.sh` — and
> [ADR-030](ADR-030-qairt-runtime-redistribution.md) records its licence terms.

## Conventions

- **One decision per record.** If a record needs a second `## Decision` heading,
  it needs a second record.
- **Status lives in front-matter** as `## Status` / `## Date` / `## Deciders`,
  followed by `## Context`, `## Decision`, `## Alternatives considered`,
  `## Consequences`, `## References`.
- **Never delete, never renumber.** Supersede, and leave a note.
- **Amendments are revision notes**, placed inside the record being amended, not
  new ADRs.
