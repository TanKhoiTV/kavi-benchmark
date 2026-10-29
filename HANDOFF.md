# Handoff — host-side track

Written 2026-10-25 as a self-contained state snapshot, refreshed 2026-10-29. If you are picking this up cold, read this file, then [`docs/README.md`](docs/README.md), then the Phase-3 requirements page.

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

Updated 2026-10-29. The gate is now a **measurement of the whole chain in one process on one clock** (staged register audio, VAD in the loop), not a sum of per-stage numbers:

| direction | n | first audio p50 | p95 | inside 2.0 s | BLEU vs organizer gold |
| --- | --- | --- | --- | --- | --- |
| VI→EN | 99 | **1607 ms** | 2011 ms | 92/99 | 22.00 |
| EN→VI | 99 | 2477 ms | — | 11/99 | 23.48 |

Those 200 organizer clips (100 `VE-*`, 100 `EV-*`) are the **calibration set**: every claim in this project is quoted against them and no change may regress them. Shipped configuration per direction: Silero VAD at `minSilenceDuration` 300 ms, **M2M-100 for MT in both directions** (greedy), 5 diffusion steps, TTS 4 threads, MT 3 threads, first clause ≥5 words.

Where the two directions diverge, same run:

| stage | VI→EN | EN→VI |
| --- | --- | --- |
| ASR | 81 ms | 71 ms |
| MT to first clause | 436 ms | 426 ms |
| TTS of first clause | **1090 ms** | **1336 ms** |

ASR and MT are the same on both sides. EN→VI misses the gate because Vietnamese TTS costs ~250 ms more per clause and Vietnamese text is longer than its English source, so **the TTS stage is the gate in both directions** — and it is also what makes beam search unaffordable today.

Peak RSS is 1258 MB against the published 1160 MB budget, down from 1667 MB once Opus-MT stopped being co-resident with M2M-100. Memory is a **low-priority gate for Round 3**, decided deliberately: the shipped checkpoints are the ones that produce the quality numbers above.

## What is settled, and what was retracted

Settled: greedy MT (beam=1), target-side token streaming for clause cutting, five diffusion steps, no contention allowance, AI Hub as a portability/quantization lab rather than a latency source.

Settled since 2026-10-27, each on a measurement:

- **One MT model, both directions: M2M-100.** 23.66 against 13.83 BLEU on ASR-shaped register input, at +268 ms and still inside the gate. One tokenizer, one prefix scheme, and Opus-MT's 211 MB leaves residency.
- **EN ASR is parakeet-tdt-0.6b-v2** (CC-BY-4.0): 12.20% against 27.83% WER on organizer EV gold, and 24.12 against 20.54 EN→VI BLEU. This is ADR-008's own risk-register fallback clause firing, not a reversal.
- **Greedy ships; beam 3 is a measured reserve.** +2.12 BLEU for 217 ms of p50 and 20 clips of gate compliance is not affordable until the TTS stage shrinks.
- **Memory is a low-priority gate.** The EN checkpoint costs +1.05 GB peak; it was accepted as a quality purchase and recorded as such rather than argued away.

Retracted, and worth not re-litigating:

- "MT slows ~20% under TTS load" — the deltas straddle zero at ±7% across processes.
- "4 diffusion steps is acceptable with post-processing" — listening rejected it; peaks destroy words.
- The isolated thread matrix ("TTS 3 threads beats 4") — measurement was right, conclusion was not transferable; under real overlap TTS wants 4.
- "Endpointing loses speech" — wrong. A timeout **splits the turn**; the remainder is still translated. Severity was overstated, and the correction is recorded in [`endpointing-gate.md`](docs/reference/endpointing-gate.md).
- "**Opus-MT is the MT**" (ADR-032's pin) — superseded in the app on the measurement above; it stays in `bench/` as the harness baseline.
- "**Endpointing is the cheapest real win.**" Against gold it is the most *expensive* latency in the budget per point of quality: 500→150 ms buys ~340 ms of hangover for +14.7 pp of first-segment WER. 300 ms is the knee.
- "**8.4% ASR WER through the shipped path.**" That was a harness bug — the replay took only the first VAD segment. With all segments concatenated it is 2.7%, against 2.0% whole-clip.
- "**Whisper is the EN fallback.**" whisper-small.en recovers 2.49 of the 6.74 BLEU oracle gap; parakeet recovers 3.58 and keeps the transducer interface already shipped.
- "**MT threads and priority are latency levers.**" The whole 2/3/4 × priority grid spans 55 ms (4%).

## Training: the lanes ran, nothing was scored

Free compute is not the constraint — 3 accounts give roughly 244-488 T4-hours before the deadline. The constraints are **data licensing** and engineering attention. The licence-clean Vietnamese speech available is ~8 validated hours of CC0 Common Voice plus FLEURS; VLSP-100h and InfoRe have never had a published licence and VIVOS is NC-SA.

State as of 2026-10-29: **eight lane definitions were built and five were trained** (`training/`, one lane per account, interleaved so one account dying does not destroy a comparison), and **none of them was ever scored**. The checkpoints are on disk under `eval_data/kaggle/output/` — about 10 GB across `mix-lr3e5-e3`, `mix-lr1e5-e3`, `pc-mix-lr1e5-e1`, `pcfix-lr1e5-e1` and `pc2-lr1e5-e1`, last written 2026-10-25/26. That directory is gitignored and regenerable only by re-running Kaggle, so it is either scored soon or deleted; it is not a durable asset.

The run itself produced four corrections to the plan rather than a result: targets must be tokenised with `spm_target` and not `spm_source`, targets must be punctuated, the source must be ASR-shaped, and target brevity has to be capped. Those are in the branch commits (`training/`), and the score-here step in [`training/README.md`](training/README.md) is still unexecuted — the corpus never leaves this machine, which is why scoring was never automated into the Kaggle kernel.

The TTS path is **ZipVoice-Distill** (Apache-2.0 base, released 4-8 step weights, runtime already in the shipped AAR) — pending a written ruling on whether Apache-2.0 weights trained on CC-BY-NC Emilia data clear ADR-029.

## Next steps, in order

1. **The submission package**, which is what the round actually scores and none of which exists: the 30-minute conversation scenario, an **installable APK plus offline instructions**, the benchmark logs the template demands, and the uncut demo video with the airplane-mode proof. Pillar D is 15%, and the brief makes the video the **primary scoring path** if a judge cannot deploy the package.
2. **Score the five fine-tune lanes that already ran.** The compute is paid for and the checkpoints are on disk; the missing half is a local scoring pass and a verdict page.
3. **Close the EN→VI gate** (2477 ms against 2000 ms). The levers with measured support are speculation on the near-end prefix (437-518 ms at 15% wasted synthesis) and, after it, the TTS step count. Anything here is gated by **no regression on the 200 calibration clips**.
4. **The TTS few-step work**, because it is the unlock for everything else: at 3 steps, beam 3 lands at 1438 ms with 93/99 inside the gate and +2.12 BLEU, so the step count is what converts beam search from a trade into an improvement.
5. **Decide the second language pair.** The number of pairs sits *inside* the 45% pillar, and the hidden benchmark is 1,000 sentences each for VI↔EN, VI↔ZH and VI↔KO.
6. **The UX levers** — pillar C is 20% and specified as worker-centred design; `kavi-android#57` holds several items that are cheap and land squarely in it.
7. **Pin the shipped models in `assets.lock.toml`.** M2M-100 is on disk and in the app but is not pinned or fetched by this repository, so the model that ships cannot be reproduced from a clean clone. Parakeet is in the same position.
8. **AI Hub**: compile for 8 Gen 3 (v75), audit offload %, run `numerics` on the MT decoder. Also the sherpa-onnx AAR behind a fetch script + SHA-256, and the ADR-029 ruling before any TTS training.

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
