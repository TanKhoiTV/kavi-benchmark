# Cold reference — historical benchmark docs

These are **no longer current** but are kept for the content they still contribute. Do not treat them as source of truth for architecture — decisions live in `../decisions/`, and the on-device records are in the private `kavi-android` repo.

| File | Why it is here | Still consulted for |
| --- | --- | --- |
| `benchmarking-plan.md` | Pre-decision dataset/candidate landscape | Corpus catalog + licenses (§3), noise-mixing recipe (§3.5), RTranslator baseline notes (§5) |
| `benchmarking-todo.md` | The v0 harness execution checklist (implemented) | Historical checklist; the on-device rows now point at `kavi-android` |
| `denoising-gate-results.md` | One-off host-side gate result | Wiener-adoption evidence for the on-device denoiser record (ADR-018) |
| `beam-sweep-host-aggregate.md` | Citable reduction of the host beam sweep (raw output is gitignored) | Quality-vs-latency evidence for open parameter #6 (MT beam width) |
| `phase3-submission-requirements.md` | Phase-3 brief and submission template read for operative requirements | Deadline, scoring pillars, mandatory items, and the gaps they expose |
| `translation-ux.md` | Review corpus of shipping translation apps, read for what users actually punish | The UX levers that cost no model budget, and the argument for recovering quality after the gate instead of before it |
| `competitive-context.md` | External context gathered after the latency design was settled | RTranslator's actual defaults, where our streaming design sits in the simultaneous-translation literature, the endpointing evidence, and the contest field |
| `endpointing-gate.md` | Test of whether the transcript can carry the endpointing decision | Why a text-only gate does not recover the VAD latency, and the corrected severity of a turn split |
| `pause-distribution.md` | Measured intra-utterance pause distribution behind `vad_speech_timeout_ms` | Why 300 ms is unsafe, why 500 ms already truncates, and why the timeout cannot be raised |

Moved out during the 2026-10 re-organisation: `phase-4-qnn-plan.md` and `rtranslator-comparison.md` → `kavi-android/docs/reference/`; `contest-info.md` and `additional-reading.md` → `aivoice-2026`. Nothing was lost, and nothing was copied twice.
