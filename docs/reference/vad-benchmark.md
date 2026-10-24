# VAD benchmark (Stage 3A, 3B, 4)

How the VAD benchmark is built, what the energy VAD does, how it is scored, and
what is an assumption. ADR-022 fixes only an amplitude threshold on the PCM float
stream and a configurable silence period (default about 500 ms); everything marked
**assumption** below is a gap-fill that needs confirmation against the Stage 0/1/1b
documents, which are not in this repository.

## Pipeline and commands

| Step | Command | Output |
|---|---|---|
| Reference speech bounds per FLEURS clip | `make vad-bounds` | `bench/vad_source_bounds.json` (committed) |
| Benchmark manifest (800 items) | `make vad-manifest` | `eval_data/vad_manifest_v2.json` (not committed) |
| Manifest and gold audit | `make vad-audit` | printed report, exit 1 on any problem |
| Render gold clips for hand labelling | `make vad-gold-audio` | `eval_data/vad_gold_audio/`, blank `eval_data/vad_gold_labels.csv` |
| Collar from hand labels | `make vad-collar` | `bench/vad_collar.json` (committed) |
| One configuration | `make vad-run THRESHOLD=0.05 TIMEOUT_MS=500` | `bench-results/vad/` |
| Smoke test, ADR defaults | `make vad-smoke` | `bench-results/vad-smoke/` |
| Full 5 x 5 sweep | `make vad-sweep` | `bench-results/vad-sweep/sweep.{json,csv}` |
| Sensitivity of the sweep to the collar (no labels) | `make vad-collar-sensitivity` | `bench-results/vad-collar-sensitivity/` |
| Stage 5: full sweep with the Stage 3B collar, verified table and record | `make vad-stage5` | `bench/results/vad/sweep_table.csv`, `sweep_record.md` (committed); raw in `bench-results/vad-stage5/` |
| Stage 6 helper: criteria applied to every pair and cell | `make vad-decision` | `bench-results/vad-decision/` |
| Borderline follow-up, 40 items per cell | `make vad-manifest-40`, then `make vad-sweep MANIFEST=eval_data/vad_manifest_v2_n20.json` | |

## Benchmark construction

- Each item is `1000 ms silence + speech A + pause + speech B + 1000 ms silence`.
  Ground truth is the A span and the B span from construction (never from RMS).
- Speech A and B are the Silero VAD speech bounds of two FLEURS clips
  (`bench/vad_source_bounds.json`, silero-vad library defaults, computed on the
  level-normalised clip). Whole clips are not used because FLEURS clips carry up to
  about 2.3 s of leading silence and long internal pauses.
- Level: each clip is scaled so the 99th percentile of its 10 ms frame peaks is 0.35
  (**assumption**). FLEURS source levels differ by about 100x, and the sweep
  thresholds are absolute. Reverberation (RIRS) can raise the level afterwards; this
  is reported, not corrected.
- Noise and RIR are applied to the whole timeline. SNR is the RMS of the
  reverberated clean timeline over the ground-truth speech segments against the RMS
  of the aligned noise over the whole timeline (**assumption**), recomputed from the
  written WAV by `scripts/validate_stage3a_representative.py`.
- Conditions and SNR: quiet (clean, +15), street (+10, +5), indoors (+10, +5),
  near-field (+5, 0), far-field (0, -5). Noise categories steady and impulsive are a
  project heuristic over MUSAN, not official MUSAN labels. near-field and far-field
  are acoustic categories, not measured distances. Unreadable RIR files are skipped.
- Pauses 200, 350, 500, 700 ms. 10 VI + 10 EN items per condition x SNR x pause cell,
  800 items. The same 20 source clips are reused in every cell.

## Energy VAD (ADR-022 gap-fills, all assumptions)

- 16 kHz mono float. 10 ms non-overlapping frames (every Stage 1 pause and timeout is
  a whole number of frames).
- Frame energy is the peak absolute sample; a frame is speech when
  `energy >= energy_threshold`. One threshold, no hysteresis.
- Hang-in is one frame. A segment closes when the run of non-speech frames is strictly
  longer than `speech_timeout_ms`, so a pause equal to the timeout never splits.
- Segment offset is the end of the last speech frame. End of speech is declared one
  frame after the timeout is exceeded; if the audio ends first, no end of speech is
  declared (`noEOS`).
- Defaults: `energy_threshold = 0.05` (Stage 1 baseline, ADR leaves it open),
  `speech_timeout_ms = 500`. No smoothing, adaptive threshold or pre-roll.

## Metrics

Scoring uses `bench/vad_scorer.py` with the collar from `bench/vad_collar.json`. Until
that file exists the scorer default of 0.150 s is used and every output says
PROVISIONAL. The scorer receives the whole A + pause + B span as one utterance.

- `jsplit` / `jviol`: the VAD did not bridge the A-B pause (`jviol` only when
  pause <= timeout). This is the rule "pause <= timeout must not split".
- `anySpl%`: any gap longer than the timeout anywhere in the utterance, including
  natural pauses inside the FLEURS sentences. It is not a test of the pause rule.
- Latency columns: `EOU` is VAD offset minus true end (negative values kept);
  `EOS` is declared end of speech minus true end. Only end-to-end turnaround is
  compared with the 2000 ms limit, and it needs ASR, MT and TTS, which this runner
  does not run.

## Collar (Stage 3B)

Reference-VAD boundaries (Silero) are compared with hand-labelled gold boundaries on
the 50 gold clips, onset and offset separately (error = reference - hand). The collar
is the 95th percentile (nearest rank) of the absolute errors pooled over onsets and
offsets (**assumption**; the confirmed Stage 1b rule takes precedence). It is compared
with the 150 ms onset tolerance and the 50 ms clipped-speech criterion and reported,
never adjusted. `make vad-collar` refuses to run unless all 50 clips are labelled.

## Collar sensitivity

The collar changes three scorer outputs: a VAD onset further than the collar from the
ground-truth onset is a missed onset, the delay beyond the collar is clipped speech,
and a VAD segment starting outside the collar and the ground truth is a false trigger.
`bench/vad_collar_sensitivity.py` scores the same VAD output at collars of 0 to
200 ms and reports the smallest collar from which every criterion is met. It shows
which conclusions depend on the collar; it does not estimate the true collar, which
only Stage 3B (hand labels) can. Collars above 150 ms exceed the Stage 1 onset
tolerance.

## Borderline rule

A metric inside the margin of its criterion is borderline: it does not count as a
pass, and the cell needs 40 items instead of 20. The criterion is never relaxed.
See `bench/vad_borderline.py` for the margins and the assumptions behind them.

## Known limitations

- Each item has 2 s of non-speech. Showing at most 0.1 false triggers per minute with
  95% confidence needs about 30 minutes of non-speech per group; the sweep prints the
  bound it can actually support.
- The 1000 ms trailing silence leaves little margin for a 1000 ms timeout.
- `bench/vad_materialize.py` keeps the last recording when a FLEURS id has
  two recordings; changing it would change the source clips and the manifest hash.
- `scipy` is imported by `bench/vad_noise.py` but not declared in `pyproject.toml`.
