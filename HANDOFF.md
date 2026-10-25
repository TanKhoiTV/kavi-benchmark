# Handoff — host-side track

Written 2026-10-25 as a self-contained state snapshot. If you are picking this up cold, read this file, then [`docs/README.md`](docs/README.md), then the Phase-3 requirements page.

## Naming: one repo, two names

The local directory is `kavi-prototype`; the GitHub remote is **`TanKhoiTV/kavi-benchmark`**. They are the same repository. Issue and PR numbers quoted anywhere in this project refer to that remote unless they say `kavi-android`.

- **This repo (public)** — host-side benchmark harness, measurements, write-ups.
- **`kavi-android` (private)** — the Android app, ADR register, QAIRT toolchain, on-device harnesses.

## Deadline

| milestone | date |
|---|---|
| **Prototype submission (Round 3)** | **13/11/2026, 23:59** |
| Judging & results | 23/11/2026 |
| Field-Testing Round | 28/11/2026 |
| Final, VNG Campus HCMC | 05/12/2026 |

The Qualcomm workshop/Q&A was 23/10/2026 — already past. Details in [`docs/reference/phase3-submission-requirements.md`](docs/reference/phase3-submission-requirements.md).

## The committed language pair

**VI↔EN only.** Breadth across VI↔ZH/KO is explicitly *not* a gap in the submission — it is a constraint to keep in mind when making decisions that would be hard to extend later.

## Where the engineering stands

The turnaround gate is **met and measured**: 1849.4 ms median, 10/10 runs inside 2.0 s, peak RSS 589 MB, on the target phone. The configuration that achieved it:

| parameter | value | why |
|---|---|---|
| VAD timeout | 500 ms | cannot move; see below |
| TTS threads | 4 | better than 3 *once MT is demoted* |
| MT threads | 2, background priority | MT's remaining decode is off the critical path |
| TTS diffusion steps | 5 | the quality line the listening tests drew |
| first spoken chunk | 4 words | shorter chunk beat every other single lever (−234 ms) |

That figure is **MT-to-first-audio measured plus 500 ms VAD and 83 ms ASR added arithmetically** — which the submission template forbids presenting as a measurement. Closing that gap is the first priority below.

## What is settled, and what was retracted

Settled: greedy MT (beam=1), target-side token streaming for clause cutting, five diffusion steps, no contention allowance, AI Hub as a portability/quantization lab rather than a latency source.

Retracted, and worth not re-litigating:

- "MT slows ~20% under TTS load" — the deltas straddle zero at ±7% across processes.
- "4 diffusion steps is acceptable with post-processing" — listening rejected it; peaks destroy words.
- The isolated thread matrix ("TTS 3 threads beats 4") — measurement was right, conclusion was not transferable; under real overlap TTS wants 4.
- "Endpointing loses speech" — wrong. A timeout **splits the turn**; the remainder is still translated. Severity was overstated, and the correction is recorded in [`endpointing-gate.md`](docs/reference/endpointing-gate.md).

## Training: decided, not started

Free compute is not the constraint — 3 accounts give roughly 244-488 T4-hours before the deadline. The constraints are **data licensing** and engineering attention. The licence-clean Vietnamese speech available is ~8 validated hours of CC0 Common Voice plus FLEURS; VLSP-100h and InfoRe have never had a published licence and VIVOS is NC-SA.

Decided order: **MT fine-tune first** (punctuation and domain, 5-20 T4-h, ships before 13/11), ASR noise second, TTS after the deadline. The TTS path is **ZipVoice-Distill** (Apache-2.0 base, released 4-8 step weights, runtime already in the shipped AAR) — pending a written ruling on whether Apache-2.0 weights trained on CC-BY-NC Emilia data clear ADR-029.

## Next steps, in order

1. **Measure the whole chain in one run** (ASR → MT → TTS, one clock, logs kept). Everything else in the submission depends on this number being a measurement rather than a sum.
2. **The 30-minute scenario** and an **installable package** — both mandatory, neither exists.
3. **MT fine-tune data plan** — **written**: [`docs/reference/mt-finetune-data-plan.md`](docs/reference/mt-finetune-data-plan.md). Its 100-item baseline withdrew the punctuation objective (the model already punctuates above the gold rate) and found verbosity and a ~400-pair authored eval set as the real requirements. **Lane set up**: [`training/`](training/README.md) builds the corpus, trains on Kaggle across four accounts (eight T4 lanes, one sweep of learning rate / schedule / length penalty / a general-only control), and scores locally. Before planning the week, run the quota probe: whether a 2xT4 hour costs one or two quota-hours decides between ~244 and ~488 available T4-hours.
4. **ADR-029 ruling on Emilia-derived ZipVoice weights** before any TTS training starts.
5. AI Hub: compile for 8 Gen 3 (v75), audit offload %, run `numerics` on the MT decoder.
6. Config follow-through: TTS thread count, and pinning the sherpa-onnx AAR behind a fetch script.

## Where the evidence lives

- **Durable:** issue comments on `kavi-android` (the numbers, tables and verdicts) and the write-ups in [`docs/reference/`](docs/reference/).
- **Generated, gitignored, not durable:** `bench-results/` and `eval_data/`. Both are reproducible from the command in the doc that cites them; nothing important exists *only* there.
- **Not redistributed:** the organizer's corpus and brief, and third-party model weights. Docs describe them and give the command, and never assume a path on someone else's machine.

## Entry points

| question | file |
|---|---|
| What does the submission have to contain, and by when? | `docs/reference/phase3-submission-requirements.md` |
| Why can't the VAD timeout change? | `docs/reference/pause-distribution.md` |
| Can training help, and in what order? | `docs/reference/training-feasibility.md` |
| What should the MT fine-tune train on, and how is it judged? | `docs/reference/mt-finetune-data-plan.md` |
| How do I launch or score a fine-tune lane? | `training/README.md` |
| Is AI Hub worth using after all? | `docs/reference/aihub-capability-review.md` |
| What do users of shipping apps actually punish? | `docs/reference/translation-ux.md` |
| How does this compare with competitors and the literature? | `docs/reference/competitive-context.md` |
| What is current vs cold reference? | `docs/README.md`, `docs/reference/README.md` |
