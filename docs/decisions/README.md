# Architecture Decision Records — public benchmark register

The register is split by what each record constrains (TanKhoiTV/kavi-benchmark#129). This directory is the **benchmark** half: it holds the decisions and open parameters whose evidence was measured here.

| Where | Records | Why there |
| --- | --- | --- |
| **Here** | ADR-032 (MT model choice) + the open-parameter rows below | The measurement stays where it was measured |
| `kavi-android/docs/decisions/` (private) | ADR-001 … ADR-028 | The on-device mechanism |
| `aivoice-2026/docs/decisions/` (public) | ADR-029 licence gate · ADR-030 QAIRT redistribution · ADR-031 Piper GPL split | Licensing is program layer |

**Rule:** keep the measurement where it was measured, and keep the decision next to the record it constrains. Where the two disagree, the public side keeps the claim and the private side the mechanism. On-device records are cited by title and repository, never by path — `ADR-020 — pipeline concurrency (kavi-android)`.

## Index

| ADR | Decision | Status |
| --- | --- | --- |
| [ADR-032](ADR-032-mt-model-choice.md) | MT model choice — Opus-MT vi↔en | Accepted. Closed TanKhoiTV/kavi-benchmark#110. VI→EN is the baseline for the app build; EN→VI is pinned unmeasured; M2M-100 is the measured challenger at +78% mean sentence-BLEU and replaces this record if the Phase-3 gate says so. |

## Open parameters

Decisions deliberately left unclosed. Rows **3 and 6** are owned publicly — their evidence is a published measurement or a program-level decision. The rest are tracked privately in `kavi-android/docs/decisions/README.md`; the numbering is shared, so a reference to "#7" means the same parameter on either side.

| # | Open parameter | Owned by | Closes when |
| --- | --- | --- | --- |
| 3 | **Piper engine licence path** | ADR-031 (`aivoice-2026/docs/decisions/`) | **Downstream of the TTS decision — not blocking.** Piper is fallback-only in the working position: dropped from the shipped stack → closes outright; kept as fallback → the MIT-era `rhasspy/piper` snapshot must be pinned and subprocess isolation adopted. |
| 6 | **MT beam width** (greedy vs beam=4) | ADR-020 (private record), measured here | **Needs the real pipeline overhead, nothing else.** Quality is settled on the guarded path — [`../reference/beam-sweep-mt.md`](../reference/beam-sweep-mt.md) finds **both BLEU definitions peaking at beam=4** (12.45 sentence / 11.91 corpus), 0.74 corpus BLEU clear of beam=8, so the earlier "beam=4 and 8 not separable" reading was itself a repetition artifact. Device turnaround is the other half: on 50 sentences the guarded MT stage is **507 ms p50 at greedy, 778 ms at beam=2, and the gate is 2.0 s for the whole pipeline**, so greedy keeps ~0.9 s of headroom while beam=2 sits on it. Closing number: measured non-MT overhead from Build C. *(TanKhoiTV/kavi-benchmark#115, #88, #160, kavi-android#36)* |

Tracked privately (`kavi-android/docs/decisions/README.md`): **#2** on-device denoiser model (ADR-018) · **#5** ASR thread-tuning build (ADR-012) · **#7** Opus-MT w8a16 vs w8a8 (ADR-003).

> **Resolved and closed:** **#1 Final MT model** — [ADR-032](ADR-032-mt-model-choice.md) pins Opus-MT vi↔en. The record is a baseline, not a quality verdict: VI→EN is measured at 9.84 mean sentence-BLEU (beam=5), EN→VI ships unmeasured, and M2M-100 measured 17.50 on the same items at 5.8× the parameters. The Phase-3 gate decides whether that record stands or is superseded.
>
> **Resolved and closed:** **#4 VAD thresholds** — the host-side noise benchmark behind this repo's [`../reference/vad-results.md`](../reference/vad-results.md) found no energy threshold and speech timeout that holds in noisy conditions, so none is fixed and a model-based VAD is the follow-up. The record and its revision note are private (ADR-022 in `kavi-android`); the collar was not hand-labelled and on-device confirmation is still pending, so the resolution is host-side only.
>
> **Resolved and closed:** **#8 QAIRT Community Edition access** — obtained; SDK `2.31.0.250130` is installed and in use (ADR-030 records its licence terms).

## Conventions

- **One decision per record.** Two `## Decision` headings in one file means two records.
- **Status lives in the record** as `## Status` / `## Date` / `## Deciders`, followed by `## Context`, `## Decision`, `## Alternatives considered`, `## Consequences`, `## References`.
- **Never delete, never renumber.** Supersede, and leave a note. Records that moved repositories keep their numbers; the private register does not renumber to fill the gaps left by ADR-029–031.
