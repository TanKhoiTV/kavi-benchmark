# ADR-034: One MT model serves both directions — M2M-100 supersedes the Opus-MT pin

## Status

Accepted — supersedes [ADR-032](ADR-032-mt-model-choice.md) for the shipped app.
ADR-032 remains the record of why Opus-MT was pinned first, and Opus-MT stays in
`bench/` as the harness baseline.

## Date

2026-10-29

## Deciders

Kavi team

## Context

ADR-032 pinned Opus-MT as the VI↔EN baseline and named its own overturn
condition in writing: *"The Phase-3 gate is the decision point: if VI→EN quality
is the binding constraint, this record should be superseded rather than
defended."* Two things then happened, and one of them was not in that record's
risk list.

**Quality: the binding constraint moved onto the input shape.** The pipeline feeds
MT text that has been lowercased, stripped of punctuation and given a terminal
mark, because that is what ASR produces. On that input the two candidates are not
close:

| model | input | register BLEU | mean hypothesis words |
| --- | --- | --- | --- |
| Opus-MT vi→en | clean, cased, punctuated | 26.05 | 9.3 |
| Opus-MT vi→en | ASR-shaped | **13.83** | 13.0 |
| M2M-100 418M | ASR-shaped | **23.66** | 7.5 |
| M2M-100 418M | clean | 28.54 | 7.9 |

M2M-100 on degraded input nearly matches Opus-MT on clean input, and produces 42%
fewer words — which is also less TTS on the critical path.

**EN→VI: the direction ADR-032 shipped unmeasured, and it is worse than
unmeasured.** Measured after that record was written: Opus-MT en→vi produces
coherent Vietnamese at 8.20 corpus BLEU on FLEURS, then **degenerates on
clause-length input** — 4 of 5 short expressions ran to the decode cap, emitting
`du du du khách khách du khách`-class output. That is a capability floor on short
input containing proper nouns, not a decode-tuning problem: `repetition_penalty`
2.0 with `no_repeat_ngram_size` 2 still hits the cap on at least one input. The
app translates clauses, so this is the condition that matters. M2M-100 on the
organizers' own register pairs: **21.51** corpus BLEU, no degeneration.

The contest scores "conversations across the number of language pairs the team
defines", and both directions of the one registered pair have to work. A model
that is strong in one direction and structurally broken in the other is not a
baseline; it is half a pipeline.

## Decision

Pin **M2M-100 418M** as the translation model for **both** directions, and stop
loading Opus-MT in the app.

- **Weights:** `facebook/m2m100_418M` at revision `55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636`,
  converted to CTranslate2 int8. Pinned with digests in
  [`../../assets.lock.toml`](../../assets.lock.toml); a fresh conversion from that
  revision is byte-identical to the shipped export, so the pin is reproducible
  rather than aspirational.
- **Licence:** MIT.
- **One artifact, two directions:** a source-language token (`__vi__` / `__en__`)
  plus a forced target prefix selects the direction. One tokenizer (BPE, shared
  vocabulary), one conversion, one memory line.
- **Decode:** greedy, as published in [`../.kavi.yaml`](../.kavi.yaml). Beam width
  is addressed in open parameter #6, not here.
- **Placement:** decoder on CPU (ADR-023). The encoder's NPU path is *not* carried
  over — see Consequences.
- **Scope:** VI↔EN. The checkpoint covers VI↔ZH and VI↔KO too, so a future pair is
  data work rather than a new model, but no such pair is claimed here.

## Alternatives considered

- **Keep Opus-MT for VI→EN and add M2M-100 for EN→VI.** Rejected: two artifacts
  resident at once (211 MB + 490 MB), two tokenizer schemes, two decode configs,
  and the worse model kept on the direction where the input-shape evidence is most
  damning. It also doubles the memory line that is already 98 MB over budget.
- **Keep Opus-MT and fix the input shape instead** (restore casing, real
  punctuation). Rejected on evidence: mechanical punctuation heuristics buy +1.2
  BLEU at best, and the particle-aware terminal mark — which ships — is worth
  +1.47. Neither closes a 9.8-BLEU gap, and both keep the EN→VI degeneration.
- **Hy-MT1.5-1.8B.** Licence-cleared but not convertible to CTranslate2, so it
  cannot be measured on this engine at all (TanKhoiTV/kavi-benchmark#91). Blocked
  on tooling, not merit.
- **NLLB-200 / SeamlessM4T.** CC-BY-NC. Disqualified by ADR-029.
- **Keep ADR-032 and record the measurements as a caveat.** Rejected: the record
  would then contradict the build, and `.kavi.yaml` is supposed to be the single
  source of truth for what ships.

## Consequences

**The QNN encoder conversion is now aimed at the wrong model.** The HTP v73
context binary, the AI Hub profile (2.03 ms encoder on NPU) and the `w8a16`
quantisation plan were all built around the Opus-MT encoder. M2M-100 has no QNN
conversion started. Two honest options, in order of cost: run the encoder on CPU
with the decoder (the measured state, 577 ms p50 MT), or convert M2M-100's encoder
and re-baseline the memory line, the profile and the quantisation record together.
This ADR does not choose; it names the work item that ADR-032's pin had already
paid for and this one has not.

**Memory grows and is accepted.** Full-stack peak RSS is 1258 MB against the
published 1160 MB budget. Memory is a low-priority gate for Round 3 by explicit
decision: the shipped checkpoints are the ones that produce the quality numbers.

**Disk grows.** 490 MB against Opus-MT's 73 MB, and the install package carries it.
This is the one cost that is user-visible at install time.

**Greedy is now a design constraint, not only a setting.** The app streams MT
tokens, and CTranslate2 rejects a per-step callback when `beam_size > 1`. So beam
quality is unavailable to the streaming path by construction until the streaming
interface changes — which is why #6's answer is entangled with the pipeline design
rather than with the metric alone.

**Opus-MT is not deleted, it is demoted.** It stays registered in `bench/` as the
host baseline, which is what makes the 13.83 vs 23.66 comparison reproducible.

**Two measurements this record makes possible, and does not contain:** COMET on
the register pairs (now in
[`../reference/mt-en-vi-baseline.md`](../reference/mt-en-vi-baseline.md)), and the
ASR oracle gap that decides whether the next spend is ASR or MT.

## References

- [`ADR-032`](ADR-032-mt-model-choice.md) — the superseded pin, and the overturn
  condition this record meets
- [`../reference/mt-en-vi-baseline.md`](../reference/mt-en-vi-baseline.md) — the
  register measurements for both directions, including COMET
- [`../reference/beam-sweep-host-aggregate.md`](../reference/beam-sweep-host-aggregate.md)
  — why the host latency column is not the shipped latency
- [`../../assets.lock.toml`](../../assets.lock.toml) — revision, digests, CT2 size
- [`../.kavi.yaml`](../.kavi.yaml) — the published pin this record makes true
- [`register-audit.md`](register-audit.md) — the supersession
  audit that found ADR-032 still advertising itself as current
- ADR-023 — *decoder on CPU* (`kavi-android`)
- ADR-029 — *shipping licence gate* (`aivoice-2026`)
- TanKhoiTV/kavi-benchmark#110 — ADR-032
- TanKhoiTV/kavi-benchmark#91 — Hy-MT conversion blocker
