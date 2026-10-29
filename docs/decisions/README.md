# Architecture Decision Records — public benchmark register

The register is split by what each record constrains (TanKhoiTV/kavi-benchmark#129). This directory is the **benchmark** half: it holds the decisions and open parameters whose evidence was measured here.

| Where | Records | Why there |
| --- | --- | --- |
| **Here** | ADR-032 (MT model choice) + the open-parameter rows below | The measurement stays where it was measured |
| `kavi-android/docs/decisions/` (private) | ADR-001 … ADR-028 | The on-device mechanism |
| `aivoice-2026/docs/decisions/` (public) | ADR-029 licence gate · ADR-030 QAIRT redistribution · ADR-031 Piper GPL split | Licensing is program layer |

**Audit:** [`register-audit.md`](register-audit.md) compares every record's status line against what the build actually does, and separates *stale-current* (says Accepted, build moved) from *stale-open* (says Proposed, already shipping). Read it before trusting any status line.

**Rule:** keep the measurement where it was measured, and keep the decision next to the record it constrains. Where the two disagree, the public side keeps the claim and the private side the mechanism. On-device records are cited by title and repository, never by path — `ADR-020 — pipeline concurrency (kavi-android)`.

## Index

| ADR | Decision | Status |
| --- | --- | --- |
| [ADR-032](ADR-032-mt-model-choice.md) | MT model choice — Opus-MT vi↔en | **Superseded 2026-10-29 by ADR-034.** Closed TanKhoiTV/kavi-benchmark#110 and named its own overturn condition; the register measurements met it. Kept as the baseline its successor is measured against. |
| [ADR-034](ADR-034-m2m-both-directions.md) | MT — one model for both directions (M2M-100) | **Accepted 2026-10-29.** Supersedes ADR-032. Opus-MT scores 13.83 BLEU on ASR-shaped input against M2M-100's 23.66, and degenerates on clause-length EN→VI input. Pinned in `assets.lock.toml`; the int8 export regenerates byte-identically. Opens the QNN-encoder question. |

## Open parameters

Decisions deliberately left unclosed. Rows **3 and 6** are owned publicly — their evidence is a published measurement or a program-level decision. The rest are tracked privately in `kavi-android/docs/decisions/README.md`; the numbering is shared, so a reference to "#7" means the same parameter on either side.

| # | Open parameter | Owned by | Closes when |
| --- | --- | --- | --- |
| 3 | **Piper engine licence path** | ADR-031 (`aivoice-2026/docs/decisions/`) | **Downstream of the TTS decision — not blocking.** Piper is fallback-only in the working position: dropped from the shipped stack → closes outright; kept as fallback → the MIT-era `rhasspy/piper` snapshot must be pinned and subprocess isolation adopted. |
| 6 | **MT beam width** (greedy vs beam=4) | ADR-020 (private record), measured here | **Resolved 2026-10-29 — greedy ships; beam is a measured reserve.** Whole pipeline, one clock, 99 organizer clips: greedy 1607 ms p50 with 92 of 99 inside 2.0 s at 22.00 BLEU, beam 3 at 1824 ms with 72 of 99 at 24.12 (+2.12 BLEU, 95% CI [+0.12, +4.35]). It reopens when the TTS step count falls, because that is what buys the 217 ms back: at 3 diffusion steps beam 3 lands at 1438 ms with 93 of 99, and an adaptive beam-on-short-transcript policy at 1359 ms with 99 of 99. The beam question is downstream of the TTS work, not of the metric. The measurement below is the earlier per-stage evidence that framed it. Quality now favours beam=4 on *both* registers: on FLEURS both BLEU definitions peak at beam=4 (12.45 sentence / 11.91 corpus, 0.74 corpus BLEU clear of beam=8, so the earlier "beam=4 and 8 not separable" reading was itself a repetition artifact), and on the organizer's 100 VI→EN pairs beam=4 beats greedy by **+3.87 mean sentence BLEU, 95% CI [+1.12, +6.78]** ([`../reference/mt-finetune-data-plan.md`](../reference/mt-finetune-data-plan.md) §1.4). Cost is now measured at three widths on the guarded path, same 50 sentences, same device: **507 ms p50 greedy, 782 ms beam=2, 1 016 ms beam=4** (repeat 1 060 ms; drift +1.6% greedy and +4.3% beam=4 across an A/B/A interleave). Against a 2.0 s whole-pipeline gate that leaves 910 ms at greedy, 639 ms at beam=2 — which is why beam=2 "sits on" the ~620 ms TTS chunk-1 fixed cost — and 401 ms at beam=4, ~220 ms short. That arithmetic is not the first-audio path (MT-to-first-clause is 74-91 ms), so the beam=4 cost lands on turn completion and on the window where MT decelerates 8-13x under TTS. The encoder is not a factor either way: **2.03 ms** on HTP v73, 0.4% of the stage ([`../reference/aihub-encoder-profile.md`](../reference/aihub-encoder-profile.md)), so the stage is decoder time. Closing number: measured non-MT overhead from Build C. *(TanKhoiTV/kavi-benchmark#115, #88, #160, kavi-android#36, kavi-android#31)* |

Tracked privately (`kavi-android/docs/decisions/README.md`): **#2** on-device denoiser model (ADR-018) · **#5** ASR thread-tuning build (ADR-012) · **#7** Opus-MT w8a16 vs w8a8 (ADR-003).

> **Resolved and closed, then superseded:** **#1 Final MT model** — [ADR-032](ADR-032-mt-model-choice.md) pinned Opus-MT vi↔en; [ADR-034](ADR-034-m2m-both-directions.md) replaced it with M2M-100 in both directions on 2026-10-29, on the register measurements that record named as its own trigger. The record is a baseline, not a quality verdict: VI→EN is measured at 9.84 mean sentence-BLEU (beam=5), EN→VI ships unmeasured, and M2M-100 measured 17.50 on the same items at 5.8× the parameters. The Phase-3 gate decides whether that record stands or is superseded.
>
> **Resolved and closed:** **#4 VAD thresholds** — the host-side noise benchmark behind this repo's [`../reference/vad-results.md`](../reference/vad-results.md) found no energy threshold and speech timeout that holds in noisy conditions, so none is fixed and a model-based VAD is the follow-up. The record and its revision note are private (ADR-022 in `kavi-android`); the collar was not hand-labelled and on-device confirmation is still pending, so the resolution is host-side only.
>
> **Resolved and closed:** **#8 QAIRT Community Edition access** — obtained; SDK `2.31.0.250130` is installed and in use (ADR-030 records its licence terms).

## Conventions

- **One decision per record.** Two `## Decision` headings in one file means two records.
- **Status lives in the record** as `## Status` / `## Date` / `## Deciders`, followed by `## Context`, `## Decision`, `## Alternatives considered`, `## Consequences`, `## References`.
- **Never delete, never renumber.** Supersede, and leave a note. Records that moved repositories keep their numbers; the private register does not renumber to fill the gaps left by ADR-029–031.
