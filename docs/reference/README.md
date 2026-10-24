# Cold reference — historical benchmark docs

These are **no longer current** but are kept for the content they still contribute. Do not treat them as source of truth for architecture — decisions live in `../decisions/`, and the on-device records are in the private `kavi-android` repo.

| File | Why it is here | Still consulted for |
| --- | --- | --- |
| `benchmarking-plan.md` | Pre-decision dataset/candidate landscape | Corpus catalog + licenses (§3), noise-mixing recipe (§3.5), RTranslator baseline notes (§5) |
| `benchmarking-todo.md` | The v0 harness execution checklist (implemented) | Historical checklist; the on-device rows now point at `kavi-android` |
| `denoising-gate-results.md` | One-off host-side gate result | Wiener-adoption evidence for the on-device denoiser record (ADR-018) |
| `beam-sweep-host-aggregate.md` | Citable reduction of the host beam sweep (raw output is gitignored) | Quality-vs-latency evidence for open parameter #6 (MT beam width) |

Moved out during the 2026-10 re-organisation: `phase-4-qnn-plan.md` and `rtranslator-comparison.md` → `kavi-android/docs/reference/`; `contest-info.md` and `additional-reading.md` → `aivoice-2026`. Nothing was lost, and nothing was copied twice.
