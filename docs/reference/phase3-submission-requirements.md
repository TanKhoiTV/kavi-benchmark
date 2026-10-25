# Phase 3 submission: what the brief and template actually require

Sources: `1V_P3_Prototype_Brief.pdf` and `[MAKE A COPY TO USE] 1V_P3_Prototype Submission_Template.pdf`
(both in `~/Downloads/2026-10-24 (2nd part)/`). This page records the requirements as operative
facts, then the nuances that are easy to miss and the gaps they expose in this project.

## Timeline (tentative, but dated)

| milestone | date |
|---|---|
| Round 3 kick-off, brief released | 14/10/2026 |
| Workshop & Q&A (Qualcomm) | 23/10/2026 — **already past** |
| Device testing at VNG Campus | TBA |
| **Submission deadline** | **13/11/2026, 23:59** |
| Judging & results | 23/11/2026 |
| Field-Testing Round | 28/11/2026 |
| Final, VNG Campus HCMC | 05/12/2026 |

## Scoring

| pillar | weight | criteria |
|---|---|---|
| A. Voice-to-voice translation quality | **45%** | semantic accuracy, clarity, naturalness; **conversations across the number of language pairs the team defines** |
| B. On-device performance | 20% | model size; inference time; peak RAM/VRAM |
| C. Device integration & UX | 20% | completeness on device; **user-centred concept for factory, construction, logistics workers** |
| D. Demo package | 15% | recorded demo, documentation, reproducibility, roadmap |
| Hardware bonus | +0.5 | optimised hardware designed by the team |

A + B + C = 85% technical group.

## Mandatory vs encouraged

**Mandatory:** real hardware (organizer-supplied or own); all inference on-device with no
API/cloud calls at demo time; a minimal interface (app/CLI/physical button) judges can operate
without running code; one prepared conversation scenario of ~30 minutes in the Round-2 use
context; at least one registered language pair tested.

**Encouraged:** the Round-2 advanced features (multi-speaker, dialects, code-switching); a clear
roadmap and statement of current limitations; final form factor (enclosure, battery, wearable) —
a dev kit or bare board is acceptable if the pipeline runs stably.

**Required for later rounds:** teams reaching Field-Testing and the Final must add more language
pairs.

## Nuances worth acting on

1. **UX is 20%, and it is specified.** Not "make it nice" — a user-centred concept for factory,
   construction and logistics workers. This is why `kavi-android#57` should not sit deferred
   indefinitely: several of its items are cheap and land squarely in criterion C.
2. **Language-pair count sits inside the 45%.** "Conversations across the number of language
   pairs defined by the team" means a single-pair demo competes against multi-pair demos for the
   same marks. The sample data supplied is VI↔EN only, but the hidden benchmark covers VI↔EN,
   VI↔ZH and VI↔KO at 1,000 sentences each.
3. **The video is the fallback scoring path.** The brief states that if judges cannot deploy the
   package, the demo video "will serve as the primary basis for scoring". The video must
   therefore be self-sufficient: network disabled on camera, uncut, with the screen or logs
   readable so processing time is visible.
4. **Estimates may not be substituted for measurements.** The template says it explicitly. Any
   number reported as a measurement must be measurable from the submitted build on the submitted
   device, with logs.
5. **Minimum requirements must be labelled tested or estimated.** "Do not imply untested
   compatibility." One device has been tested.
6. **Model size must be distinguished from installation size**, and peak memory reported with the
   measurement tool and run conditions.
7. **Development tools must be distinguished from the submitted runtime.** AI Hub and QDC
   profiling is allowed as evidence of development; it is not the delivered path.
8. **Naming and packaging are prescribed:** `[TeamName]_Prototype Submission_V3.docx` or `.pdf`,
   plus the offline package via Drive/GitHub, the public source repo link, and the benchmark logs.
9. **Supported hardware is broader than this project's target.** The brief lists RB3 Gen 2,
   Arduino Ventuno Q, IQ-9075, Snapdragon 8 Gen 2/3+ mobile, and Snapdragon X Elite. QNN context
   binaries are silicon-specific, so a v73 (SM8550) binary will not load on the other boards.

## Gaps this exposes

| gap | why it matters | action |
|---|---|---|
| The full chain has never been measured end to end | 1849 ms is a measurement of MT→first audio plus an arithmetic addition of 500 ms VAD and 83 ms ASR, which is exactly the "estimate substituted for a measurement" the template forbids | measure ASR→MT→TTS in one process, in one run, with logs |
| Only one language pair | pair count is scored inside the 45% | decide whether VI↔EN alone is the honest submission, or add a second pair |
| Hidden benchmark is 1,000 sentences/pair | MT quality was measured on 50 items | run the larger eval and report BLEU with the metric named |
| Only one device tested | minimum requirements must be labelled | state SM8550 tested; everything else estimated |
| No 30-minute scenario rehearsal | a mandatory deliverable, and criterion C asks whether the scenario completed | build and run the script |
| No packaged installable build | mandatory, and it is the fallback path to the video | produce the APK + instructions |

## The flow-matching decks

The same folder holds `0_hook`, `1_NormalizingFlows`, `2_ContinuousNormalizedFlows` and
`3_FlowMatching` slide decks. They are general lecture material on normalising flows, continuous
normalising flows and flow matching — not project-specific. They are relevant background for the
TTS step-count work: Supertonic's vector estimator is a flow-matching model, which is why cutting
diffusion steps trades quality against latency, and why a few-step (distilled) model is the only
structural way out of that trade.
