# Architecture Decision Records — public benchmark register

The register is split by what each record constrains (TanKhoiTV/kavi-benchmark#129).
This directory is the **benchmark** half: it holds the decisions and open
parameters whose evidence was measured here.

| Where | Records | Why there |
| --- | --- | --- |
| **Here** | ADR-032 (MT model choice, pending) + the open-parameter rows below | The measurement stays where it was measured |
| `kavi-android/docs/decisions/` (private) | ADR-001 … ADR-028 | The on-device mechanism |
| `aivoice-2026/docs/decisions/` (public) | ADR-029 licence gate · ADR-030 QAIRT redistribution · ADR-031 Piper GPL split | Licensing is program layer |

**Rule:** keep the measurement where it was measured, and keep the decision next
to the record it constrains. Where the two disagree, the public side keeps the
claim and the private side the mechanism. On-device records are cited by title
and repository, never by path — `ADR-020 — pipeline concurrency (kavi-android)`.

## Index

| ADR | Decision | Status |
| --- | --- | --- |
| ADR-032 | MT model choice — Opus-MT vi→en | **Not yet written.** Tracked by TanKhoiTV/kavi-benchmark#110. Until it lands, `mt.model` in `../.kavi.yaml` records the value in use and the results in `../benchmark/` record what it scored. |

## Open parameters

Decisions deliberately left unclosed. Rows **1, 3 and 6** are owned publicly —
their evidence is a published measurement or a program-level decision. The rest
are tracked privately in `kavi-android/docs/decisions/README.md`; the numbering is
shared, so a reference to "#7" means the same parameter on either side.

| # | Open parameter | Owned by | Closes when |
| --- | --- | --- | --- |
| 1 | **Final MT model** | ADR-032 (this repo, pending) | An MT decision ADR is written (TanKhoiTV/kavi-benchmark#110). Opus-MT is in use and pinned in `../.kavi.yaml`, but the choice was never recorded. *Not measurement-gated: it needs a record, not a number.* |
| 3 | **Piper engine licence path** | ADR-031 (`aivoice-2026/docs/decisions/`) | **Downstream of the TTS decision — not blocking.** Piper is fallback-only in the working position: dropped from the shipped stack → closes outright; kept as fallback → the MIT-era `rhasspy/piper` snapshot must be pinned and subprocess isolation adopted. |
| 6 | **MT beam width** (greedy vs beam=4) | ADR-020 (private record), measured here | **Needs a number:** Phase-5 on-device latency-vs-BLEU measurement. The host-side quality half is already citable in [`../reference/beam-sweep-host-aggregate.md`](../reference/beam-sweep-host-aggregate.md) and in the published results. *(TanKhoiTV/kavi-benchmark#115)* |

Tracked privately (`kavi-android/docs/decisions/README.md`): **#2** on-device
denoiser model (ADR-018) · **#4** VAD thresholds (ADR-022) · **#5** ASR
thread-tuning build (ADR-012) · **#7** Opus-MT w8a16 vs w8a8 (ADR-003).

> **Resolved and closed:** **#8 QAIRT Community Edition access** — obtained;
> SDK `2.31.0.250130` is installed and in use (ADR-030 records its licence terms).

## Conventions

- **One decision per record.** Two `## Decision` headings in one file means two
  records.
- **Status lives in the record** as `## Status` / `## Date` / `## Deciders`,
  followed by `## Context`, `## Decision`, `## Alternatives considered`,
  `## Consequences`, `## References`.
- **Never delete, never renumber.** Supersede, and leave a note. Records that
  moved repositories keep their numbers; the private register does not renumber
  to fill the gaps left by ADR-029–031.
