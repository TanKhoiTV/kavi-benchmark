# Documentation

Internal documentation for the Kavi prototype. **Public** contest docs
(`contest-info.md`, `specifications.md`) live in the parent repo
`aivoice-2026/docs/` (the prototype copy of contest-info is deprecated — see
`reference/contest-info.md`). The registration checklist, Luma registration
answers and pitch deck were removed from both repos; they exist in no current
tree.

## Current — read these first

- `onboarding.md` — new-member guide: decodes the jargon, the ADRs, and the
  repo layout. **Start here.**
- `onboarding-quiz.md` — comprehension checkpoint (12 questions).
- `android-implementation-plan.md` — implementation plan, **milestone view**:
  M0–M6, risks R1–R5, execution order.
- `android-kotlin-cpp-implementation-plan.md` — implementation plan, **file-level
  view** (Kotlin/C++ file plan, JNI contract, Build A–F sequencing,
  `.kavi.yaml` wiring). Companion to the milestone plan above.
- `ndk-conversion-runbook.md` — step-by-step NDK + QAIRT runbook for the one
  remaining QNN artifact (Opus-MT encoder → HTP v73 context binary).
- `rtranslator-test-protocol.md` — Phase-5 APK test procedure (live).
- `issue-116-quantization-ab-plan.md` — Opus-MT encoder w8a16 vs w8a8 A/B
  (closes ADR-003 open parameter #7). Phases P0–P7, decision rule, and the
  pre-requisite gap analysis. Gated on Milestone 3.
- `ndk-conversion-runbook.md` — step-by-step NDK/QAIRT conversion runbook
  (Opus-MT encoder → HTP v73 context binary); required reading for the plan above.
- `specifications.md` — the six objective metrics + hard thresholds; cited by
  every plan as the gate contract.

## Hot reference — source of truth

- `decisions/` — Architecture Decision Records, one decision per record
  (`ADR-001` … `ADR-031`). Start at `decisions/README.md` — the index with the
  supersession graph and the open parameters. Immutable by convention; status
  changes are recorded in the ADR header.

- `benchmark/` — published v1 results: `benchmark-results-fleurs-vivos.md` (tables
  with per-candidate notes) and `report/report.pdf` (the Typst report, laid out
  per `report-layout.md`). The `.md` tables and `report/charts/*.png` are
  regenerated from `benchmark/raw-results/*.csv` by `bench/scripts/`; the raw CSVs
  are generated from `run_results.json` by `bench/scripts/results_to_csv.py`, which
  records the beam width the run results themselves do not carry.

## Cold reference — superseded / deprecated

- `reference/` — everything no longer current, kept for history and for the
  content still consulted:
  - `benchmarking-plan.md` — pre-decision dataset/candidate landscape; the
    **corpus catalog (§3) and licensing notes** are still consulted for eval work.
  - `benchmarking-todo.md` — v0 harness execution checklist (implemented).
  - `phase-4-qnn-plan.md` — QNN conversion spec; **only the Opus-MT encoder
    path (§2, §8, §9) is live**; Whisper/Piper sections are reference-only.
  - `denoising-gate-results.md` — Phase-6 gate result (Wiener ADOPTED).
  - `contest-info.md` — stale copy; canonical version is in the parent repo.
  - `additional-reading.md` — reading list.
  - `rtranslator-comparison.md` — RTranslator 2.1.5 architectural comparison
    (relocated out of `decisions/`; records no decision).

## Planned / deferred

- `architecture.md` — system architecture (deferred; the inference architecture
  is now recorded across ADR-007 plus ADR-013–ADR-022 — see
  `decisions/README.md`).
- Legacy ADRs (`001`, `002`) are in `../archive/docs/adr/`.

## Reference

The full previous documentation set is preserved under `../archive/docs/`.
