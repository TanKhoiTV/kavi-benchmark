# ADR-022: Energy-based VAD with speech timeout

## Status

Accepted — noise benchmark (host-side, collar not measured) found no energy threshold and
speech timeout that meets the criteria in the noisy conditions: **model-based VAD required**
via the reserved model slot (follow-up pending); no threshold or timeout fixed;
on-device confirmation pending

## Date

2026-07-26

## Deciders

Kavi team

## Context

The pipeline needs utterance boundaries: when to start capturing and when to
declare the end of speech. Options range from an energy threshold on the already
captured PCM buffer (near-zero overhead) to a model-based VAD such as Silero
(~5–10 ms per frame).

The choice interacts with the denoising stage
([ADR-018](ADR-018-denoising-slot.md)): if background noise has as much energy as
speech, a purely amplitude-based detector will fail.

> **Split note:** extracted from the former `ADR-007 Decision 11`.

## Decision

Use an **energy-based VAD built into the `Recorder` class** for v1, with a model
slot reserved for later.

- **Energy-based VAD:** amplitude threshold on the PCM float stream.
- **Speech timeout:** configurable silence period (default ~500 ms) triggers
  utterance-finalisation.
- **Rationale:** the energy-based approach adds ~0 ms overhead (it operates on the
  already-captured buffer). A model-based VAD (e.g. Silero) adds ~5–10 ms per
  frame.
- **Model slot reserved:** if benchmark results show energy-based VAD is inadequate
  for the target noise environments, a lightweight model can be slotted in **without
  changing the pipeline architecture**.
- **Resolved by the noise benchmark:** no energy threshold and speech timeout in the
  tested grid meets the criteria in the noisy conditions, so none is fixed. See the
  revision note below. The default timeout above is a configuration default, not a
  benchmark-validated value.

## Consequences

### Positive

- **Energy-based VAD** keeps the capture path simple and zero-overhead for v1; a
  model-based VAD can be slotted in later without pipeline changes.

### Negative / risk

- **Energy VAD may miss speech onsets in high-noise environments** — if contest
  evaluation uses noisy samples near 0 dB SNR, a model-based VAD may be required.
  The noise benchmark confirmed this on the host (see the revision note).
  Mitigation: the VAD model slot is in the pipeline; the energy threshold can be
  tuned or replaced without changing `Recorder`'s interface.

## Revision note — noise benchmark outcome (Case B)

### Method

A host-side benchmark scored an energy VAD on constructed audio. Each item is leading
silence, one FLEURS utterance, a pause, a second FLEURS utterance and trailing
silence; the ground truth comes from the construction. The speech bounds of each
source clip come from a reference VAD (Silero), not from listeners. Speech level is
normalised, MUSAN noise (steady and impulsive are a project heuristic, not MUSAN
labels) and real RIRS impulse responses are applied to the whole item at the stated
SNR, and the VAD is scored by `bench/vad_scorer.py`. The frame length, energy
definition, hysteresis, hang-in and hang-out are not fixed by this ADR; the values
used and every other assumption are in [`docs/vad-benchmark.md`](../vad-benchmark.md).
The benchmark used only a single threshold with no smoothing or adaptation.

### Conditions

| Condition | SNR levels (dB) |
| --- | --- |
| quiet | clean, 15.0 |
| street | 10.0, 5.0 |
| indoors | 10.0, 5.0 |
| near-field | 5.0, 0.0 |
| far-field | 0.0, -5.0 |

Pause levels are 200, 350, 500 and 700 ms. Each condition/SNR/pause cell has 20
items. near-field and far-field are acoustic categories, not measured distances.

### Grid and results

Energy thresholds 0.01, 0.02, 0.05, 0.1 and 0.2 were crossed with speech timeouts of
200, 350, 500, 750 and 1000 ms: 1000 rows over 40 cells. The compact results table
is [`bench/results/vad/sweep_table.csv`](../../bench/results/vad/sweep_table.csv) and
the exact command, git commit, seed, manifest hash and verification are in
[`bench/results/vad/sweep_record.md`](../../bench/results/vad/sweep_record.md). The
criteria applied are the missed-onset rate, the false-trigger rate and the clipped
speech per utterance, with the limits and borderline margins encoded in
`bench/vad_borderline.py` (to be confirmed against the Stage 1 criteria).

**Finding (Case B):** no (threshold, timeout) pair passes in every cell. Only the
quiet/clean cells pass, at thresholds 0.01, 0.02 and 0.05 for every timeout; every
cell fails at thresholds 0.1 and 0.2. Every other condition fails, including
quiet/15.0:

| Condition / SNR | `missed_onset_rate` at threshold 0.05, timeout 500 | `false_trigger_rate_per_min` at threshold 0.05, timeout 500 (rounded) |
| --- | --- | --- |
| quiet / clean | 0.0 | 0.0 |
| quiet / 15.0 | 0.1 | 4.5 to 7.5 |
| street / 10.0 | 0.65 | 25.5 to 28.5 |
| street / 5.0 | 1.0 | 37.5 to 39.0 |
| indoors / 10.0 | 0.7 | 21.0 |
| indoors / 5.0 | 0.7 | 21.0 |
| near-field / 5.0 | 0.7 | 21.0 |
| near-field / 0.0 | 0.7 | 21.0 |
| far-field / 0.0 | 1.0 | 34.5 to 39.0 |
| far-field / -5.0 | 1.0 | 34.5 to 39.0 |

In the street, indoors, near-field and far-field cells the VAD often never closes:
`eou_delay_p95_ms` is at or near 1000.0, the trailing silence of the item, so the value
is not the cost of a timeout. The pause rule (a pause no longer than the timeout must not
split) is not part of the pass/fail criteria above; `junction_split_rate` in the
table reports it.

### Label error and collar

The hand-labelled gold step (Stage 3B) was **not performed**. The label error is not
measured. The runs used a collar of 0.15 s, the scorer default, recorded as
PROVISIONAL in the command record. Missed-onset and clipped-speech figures, and the
verdict for the quiet/clean cells, depend on the collar and are provisional. The Case B
finding for the noisy conditions does not depend on it: a larger collar never worsens
the scores and 0.15 s is taken as the largest collar the onset criterion allows (see
the collar sensitivity section of [`docs/vad-benchmark.md`](../vad-benchmark.md)).

### Added end-of-utterance delay against the 2.0 s gate

No timeout is adopted, so this ADR claims no delay for one. The benchmark's
end-of-utterance delay values cannot stand in for it (see above). Any VAD selected
by the follow-up still needs an end-of-speech silence period, and its contribution
must be budgeted against the end-of-speech to speech-out gate
([ADR-020](ADR-020-pipeline-concurrency.md), `timing.turnaround_gate_sec` in
`.kavi.yaml`) when it is chosen. `.kavi.yaml` and the Android implementation plan are
unchanged because no value changed.

### Limits of the evidence

- **Host-side only.** This is not an on-device number. On-device confirmation is
  pending.
- Microphone, codec and capture-path differences are not modelled.
- Conditions not covered: speakers and sentences beyond the source clips (20 clips
  recur in every cell), noise beyond the MUSAN files and impulse responses assigned
  to the items, overlapping speech, SNR outside the grid.
- The reference boundaries come from a model, not from listeners, and the label error
  is not measured.
- The criteria and several definitions (level, SNR, borderline margins) are
  assumptions pending a check against the Stage 1 documents.
- Each cell has 20 items, so every rate is a multiple of 0.05.

### Follow-up

A model-based VAD is to be evaluated for the reserved slot on the same manifest and
scorer (follow-up issue to be created). The hand-labelled collar (Stage 3B) and
on-device confirmation remain open.

## References

- [ADR-007](ADR-007-translation-service.md) — TranslationService (parent record)
- [ADR-018](ADR-018-denoising-slot.md) — Denoising slot (the next stage)
- [ADR-012](ADR-012-asr-thread-tuning.md) — ASR thread tuning (capture-thread priority interaction)
- [`docs/vad-benchmark.md`](../vad-benchmark.md) — benchmark construction and assumptions
- [`docs/vad-results.md`](../vad-results.md) — results, limits and open items
- [`bench/results/vad/sweep_table.csv`](../../bench/results/vad/sweep_table.csv) and
  [`sweep_record.md`](../../bench/results/vad/sweep_record.md) — Stage 5 table and command record
