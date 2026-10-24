# ADR-032: Pin Opus-MT vi↔en as the translation baseline

## Status

Accepted — VI→EN is the baseline for the Phase-3 app build. EN→VI is pinned
**unmeasured**. M2M-100 is recorded as the measured challenger and replaces this
record if the Phase-3 gate says so.

## Date

2026-10-24

## Deciders

Kavi team

## Context

MT is the only pipeline stage with no decision record. ADR-004 deferred the stack
and was withdrawn; ADR-008 and ADR-009 closed only ASR and TTS. The gap is tracked
as open parameter #1 in [`README.md`](README.md) and as
TanKhoiTV/kavi-benchmark#110.

The app build now needs a pinned MT model. Three things depend on the pin: the QNN
encoder conversion (M3 deliverable), the tokenizer bundling, and the memory line
items in `.kavi.yaml`.

Any candidate has to clear five constraints:

- **Licence:** Apache-2.0 or MIT. The shipping gate (ADR-029 — *shipping licence
  gate*, `aivoice-2026`) blocks anything non-commercial, and Kavi ships a closed
  binary.
- **Directions:** VI→EN and EN→VI at minimum. Phase 3 scores VI↔ZH and VI↔KO as
  well.
- **Footprint:** MT has 126 MB of the 1160 MB budget — 46 MB encoder, 80 MB
  decoder (`../.kavi.yaml`).
- **Offline:** no network call at runtime, so tokenizers ship with the weights.
- **Convertibility:** CTranslate2 for the host benchmark, QNN for the device. The
  encoder runs on HTP v73 and the decoder on CPU (ADR-023).

## Decision

Pin **Opus-MT** as the VI↔EN translation baseline, in both directions.

- **Weights:** `Helsinki-NLP/opus-mt-vi-en` at revision `c8d2853e`, plus
  `Helsinki-NLP/opus-mt-en-vi`. Two checkpoints, one architecture, one conversion
  path, one tokenizer scheme (`source.spm` + `target.spm`).
- **Licence:** Apache-2.0, both checkpoints.
- **Decode:** greedy, as `../.kavi.yaml` publishes. Beam width stays open
  parameter #6 and is not resolved here.
- **Placement:** encoder on HTP v73 at `w8a16`, decoder on CPU (ADR-023).
- **Scope:** VI↔EN only. VI↔ZH and VI↔KO are outside this pin — no Opus-MT
  checkpoint exists for either.
- **Why now:** the encoder is 186 MB fp32 and the decoder 322 MB fp32. Both need
  conversion and quantisation before the app can translate, and only Opus-MT has
  that path started.

The decision is a baseline, not a verdict on quality. §Consequences names the
number that would overturn it.

## Alternatives considered

### M2M-100 418M

MIT licence. Covers Vietnamese, English, Chinese and Korean from one checkpoint,
which is the only candidate that covers all four Phase-3 directions.

**Measured on the same 347 FLEURS items** (`../benchmark/benchmark-results-fleurs-vivos.md`):

| candidate | mean sentence-BLEU | mean latency | peak RAM |
| --- | --- | --- | --- |
| m2m100-vi-en-ct2-cpu | **17.50** | **0.97 s** | 1102 MB |
| opus-mt-vi-en-ct2-cpu | 9.84 | 1.64 s | 399 MB |

M2M-100 wins quality by 78% and wins host latency outright. It loses on weight:
about 5.8× the parameters, and 1102 MB of harness peak RAM against 399 MB.

**Rejected for now, not for good.** No QNN conversion exists, and no evidence yet
that a 418M autoregressive decoder stays inside the 2.0 s turnaround gate with the
decoder on CPU. The reference for that gate is an on-device number that nobody has
measured. See §Consequences for the trigger.

### Hy-MT1.5-1.8B

Licence cleared — the regional carve-out permits a commercial product in Vietnam
(ADR-029). It cannot be converted to CTranslate2 int8, so it cannot be benchmarked
on the current engine (TanKhoiTV/kavi-benchmark#91). Blocked on tooling, not on
merit.

### MADLAD-400

Apache-2.0 and clean on licence. Blocked on size, which the licence does not fix.

### NLLB-200-distilled-600M and SeamlessM4T v2

CC-BY-NC-4.0. Disqualified by the shipping gate. No measurement needed.

### No record

Rejected. `.kavi.yaml` already claims an MT model, and a claim without a record
leaves the next engineer to re-derive the choice from results alone.

## Consequences

**EN→VI ships unmeasured, and that is the main risk in this pin.** No EN→VI run
exists, and the organizers' EN→VI sample carries no reference transcript to score
against. The pin is symmetric because the engine is, not because the quality is.
First task after this record: score EN→VI on the FLEURS pairs. A large asymmetry
between the directions would reopen a decision this record currently closes.

**The published number is better than the shipped config.** The 9.84 comes from
beam=5. Greedy scored 6.22 against 9.01 at beam=5 on the 30-item beam sample
(`../reference/beam-sweep-host-aggregate.md`). Pinning greedy therefore gives up
roughly a third of the quality the benchmark reflects. Open parameter #6 decides
whether that trade stands.

**9.84 is not "Opus-MT translates badly".** That figure is a mean of per-sentence
BLEU, not corpus BLEU, so it does not compare to published BLEU for this model.
The ratio between the two candidates is the citable part, because both were
measured the same way on the same items.

**M2M-100 is the named successor.** The cheapest moment to swap is now, before M3
produces the context binary. After that, a swap re-baselines the encoder
conversion, the memory budget and the MT gate together. The Phase-3 gate is the
decision point: if VI→EN quality is the binding constraint, this record should be
superseded rather than defended.

**A second engine is already likely.** Opus-MT ships no VI→ZH checkpoint (only
ZH→VI) and no Korean at all. Its lightness argues for keeping it for VI↔EN, not
for stretching it to pairs it does not have.

**Adding a pair is data work, not a schema change.** The manifest schema already
carries `direction`. A new pair needs a checkpoint, one `assets.lock.toml` entry, a
candidate id, and a `reference_text` per item.

**This closes open parameter #1.** It does not touch #6 (beam width) or #7
(`w8a16` vs `w8a8`).

## References

- [`../.kavi.yaml`](../.kavi.yaml) — `mt.model`, decode config, memory budget
- [`../../assets.lock.toml`](../../assets.lock.toml) — pinned revision, digests, CT2 size
- [`../benchmark/benchmark-results-fleurs-vivos.md`](../benchmark/benchmark-results-fleurs-vivos.md) — the BLEU, latency and RAM table
- [`../reference/beam-sweep-host-aggregate.md`](../reference/beam-sweep-host-aggregate.md) — greedy vs beam, and why the host latency column is not citable
- ADR-023 — *encoder on NPU, decoder on CPU* (`kavi-android`)
- ADR-020 — *pipeline concurrency* (`kavi-android`), holds the beam-width deferral
- ADR-029 — *shipping licence gate* (`aivoice-2026`)
- TanKhoiTV/kavi-benchmark#110 — this record
- TanKhoiTV/kavi-benchmark#91 — Hy-MT conversion blocker
- TanKhoiTV/kavi-benchmark#115 — beam-width measurement
