# Decision register audit — 2026-10-29

A record whose status line contradicts the build is worse than no record, because it is read as an instruction. This page compares every record's status against what the app actually does and what has been measured since, and separates the two ways a register rots.

- **Stale-current** — the status says *Accepted*, but the build has moved past it.
- **Stale-open** — the status says *Proposed* or *open*, but the thing is already shipping or already measured.

Scope: the public register here, the on-device register in `kavi-android` (cited by number and title, never by path), and the program register in `aivoice-2026`. Method: status line against the build, then against the measurement pages in [`../reference/`](../reference/). Nothing below changes a record I do not own — the actions column says which are done and which need an owner.

## What the build does today

This is the reference point every status line is judged against, and it is deliberately blunt: **there is no app UI**, so "ships" means the code path every measurement runs through.

| stage | what runs | record that is supposed to describe it |
| --- | --- | --- |
| ASR, VI | Apache-2.0 icefall zipformer transducer (`zzasdf/viet_iter3_pseudo_label`), 2.0-2.7% WER on the register | ADR-008 |
| ASR, EN | **parakeet-tdt-0.6b-v2** (CC-BY-4.0), 12.20% on organizer gold | ADR-008 |
| direction | **explicit `Direction` enum**, chosen by the caller — no language detection anywhere in `app/src/main` | ADR-008 → ADR-033 |
| MT | **M2M-100 418M int8**, both directions, greedy, source token + target prefix | ADR-032 → **ADR-034** |
| TTS | Supertonic, **5 diffusion steps** | ADR-009, ADR-028 |
| VAD | **Silero `ModelVad`**, `minSilenceDuration` 300 ms | ADR-022 |
| encode/decode placement | MT encoder and decoder both on **CPU** today | ADR-023 |
| measured chain | VI→EN 1607 ms p50 / 22.00 BLEU; EN→VI 2477 ms / 23.48 BLEU | — |

## Findings

| record | its status line says | the build and the measurements say | action |
| --- | --- | --- | --- |
| **ADR-008** *v1 Android ASR — dual Zipformer* | Accepted, with the language-detection half set aside by ADR-033 | Only the VI leg is a zipformer. The EN leg has been parakeet-tdt since 2026-10-29, on the record's **own** risk-register fallback clause: 27.83% → 12.20% WER on organizer gold | **stale-current.** Needs a revision note naming the EN swap; the "VAD/ASR interface" half of the record is still accurate |
| **ADR-009** *v1 Android TTS — SupertonicTTS 3* | Proposed | Shipped: every measurement since July runs it, at 5 steps | **stale-open.** Accepted, with the step count named as the open lever |
| **ADR-010** *`ALL_OPT` is safe* | Proposed | In use and unchanged | **stale-open.** Accept |
| **ADR-012** *ASR thread tuning* | Proposed | Thread counts are measured and closed — for MT (55 ms across 2/3/4 × priority) and TTS (4 optimal, more hurts). ASR itself is 71-91 ms of the chain and is no longer worth tuning | **stale-open, and lower value than it looks.** Accept or withdraw; the latency is not there |
| **ADR-016** *Peak memory budget* | Accepted, revised 2026-07-27 | **1258 MB measured against the 1160 MB budget**, and the overrun was accepted deliberately when the EN checkpoint changed | **stale-current.** Needs a revision note; the budget line is now a historical target |
| **ADR-018** *Tiered denoising slot* | Accepted, model pick **OPEN** | The host-side gate measured the Wiener reference on constructed noise and it did not pay at the operating point | **stale-open.** The model pick can close as "no denoiser in v1, slot stays"; leaving it open implies work is pending |
| **ADR-022** *Energy-based VAD with speech timeout* | Accepted — the noise benchmark found **no energy threshold or timeout that meets the criteria** and concluded *"model-based VAD required, follow-up pending"* | The follow-up happened without a record: every measurement since 2026-10-27 runs **Silero `ModelVad`**, and the 300 ms figure is a model-VAD number | **stale-followup, not stale-current.** Corrected from an earlier reading of this audit: the record is honest about the *conclusion* and silent about the *adoption*, which is a different and smaller defect — the register points at a slot with no decision behind it, rather than at a detector we do not run |
| **ADR-023** *Encoder on NPU, decoder on CPU* | Accepted | Both are on CPU: the QNN encoder path was built for the **Opus-MT** encoder, and the shipping encoder is M2M-100's | **stale-current, and now a work item.** ADR-034 records the consequence rather than hiding it |
| **ADR-024** *Piper TTS stays on CPU* | Accepted | Piper is fallback-only; Supertonic is what runs | **harmless but historical.** Keep as the fallback's placement record |
| **ADR-028** *Migrate v1 TTS to VieNeu-TTS (Phase 2)* | Proposed | Not started; the live TTS lever is the step count, and ZipVoice-Distill is the candidate under ADR-029 | **stale-open.** Either schedule it or move it behind the deadline explicitly |
| **ADR-033** *Language detection — the dual-ASR confidence method is not readable in the shipped runtime* | Proposed | Correct and unbuilt. Host-side separability is measured (97.0% correct direction, both directions, clean audio); **no detection code exists in the app** | **stale-open, and now prioritised** — automatic detection is a requirement for the v2 build, not a nice-to-have |
| **ADR-032** *Pin Opus-MT as the translation baseline* | Accepted | Superseded: both directions ship on M2M-100 | **fixed today** — marked superseded by [ADR-034](ADR-034-m2m-both-directions.md) |
| `kavi-tuning.yaml` *TTS steps* | `num_steps: 8` | Every documented measurement uses **5** | **config drift.** The file that claims to be the mechanism contradicts the number in every table |

## The view I would hold

**About eight records are load-bearing for v2; the rest are history.** The set that actually constrains the next build: ADR-002 (platform), ADR-011 (asset provenance), ADR-013/ADR-016 (residency and memory, as measured rather than as budgeted), ADR-019 (runtime bundling), ADR-023 (placement, now open again), ADR-027 (ASR on CPU), ADR-029 (licence gate), ADR-033 (detection), ADR-034 (MT). Everything else either describes a component that was replaced or records a choice whose cost has since been paid.

**Three structural fixes, rather than twelve status edits:**

1. **The VAD needs a record for the model it actually runs.** ADR-022 correctly refuted the energy threshold and left the model slot open; Silero was then adopted without a record. A model-VAD record — the 300 ms figure, the 60-90 ms offset between energy end and detector end, and the phantom-segment behaviour the on-device harness invariants record (INV-4, `kavi-android`) — would close a slot that currently has no decision behind it.
2. **ASR and language detection should be separate records, and now are.** ADR-008's own status admits the split; ADR-033 carries the detection half. The remaining work is to make ADR-008 say *two single-language transducers* and stop claiming one architecture for both legs.
3. **TTS needs one record with the step count in it.** ADR-009 (Supertonic, Proposed) and ADR-028 (VieNeu, Phase 2) both describe the same slot at different times, and neither states the number that decides the latency budget. The step count is the highest-value lever left, so it should be in a record, not in a config file that disagrees with the measurements.

**And one rule to stop the rot**, which is cheap: **a status line names the evidence that last moved it, with a date.** "Accepted — revised 2026-07-27 by ADR-008" is what a healthy line looks like; a bare "Accepted" on a record from July means nobody has re-read it since.

## Not mine to close

- **ADR-023 / the encoder:** does M2M-100 get a QNN encoder conversion, or does the encoder stay on CPU and the AI Hub profile become a historical artifact? This is the largest unpaid bill in the register.
- **ADR-022's follow-up:** is Silero the accepted model VAD (and therefore a record), or is the slot still open with Silero as a stop-gap? Either answer needs writing down.
- **ADR-009/028:** which TTS ships for the final, and at what step count? The answer decides whether beam search is affordable at all.
- **ADR-012 and ADR-018:** close them or delete the intent, but leaving them *Proposed* implies work nobody is doing.

## What this audit changed today

[ADR-034](ADR-034-m2m-both-directions.md) written; ADR-032 marked superseded; M2M-100 pinned in [`../../assets.lock.toml`](../../assets.lock.toml) with digests that regenerate byte-identically; open parameter #6 closed on the measured whole-pipeline numbers. The rest is recorded here rather than edited, because most of those records are private and their mechanism is not mine to rewrite from the public side.

## Update — the same day, after this audit was merged

The private register was repaired where the finding was a fact about the build rather than a decision to be made: ADR-008 now records the English leg's move to `parakeet-tdt-0.6b-v2`, ADR-009 and ADR-010 moved from *Proposed* to *Accepted* (both have shipped since July), and ADR-016 records 1258 MB measured against its 1160 MB budget. ADR-033 moved to *Accepted* and became release-blocking, because **ADR-035 — *direction is detected, never asked*** (`kavi-android`) withdraws the push-to-talk fallback that ADR-033 deferred to.

That leaves two records on the "not mine to close" list rather than four: ADR-022's follow-up (does a model-VAD record exist or is the slot open?), and ADR-023's encoder placement, which ADR-034 reopens by shipping an encoder with no QNN path.
