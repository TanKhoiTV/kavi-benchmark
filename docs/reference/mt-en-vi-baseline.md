# MT baseline — EN→VI, the direction that had never been measured

**Measured** 2026-10-27 on the host (CTranslate2 4.8.0, CPU, int8) over 200 FLEURS
`en_us`→`vi_vn` parallel pairs (`eval_data/mt_en_vi_eval_manifest.json`, built with
`bench.data_prep.build_mt_manifest(direction="en->vi")`).

ADR-032 pinned Opus-MT `en→vi` for the app but recorded the leg as **"pinned
unmeasured"**. This closes that: the direction works, and it is the same strength as
the other one.

| decode | mean sentence BLEU | corpus BLEU | outputs with repeated trigrams |
| --- | --- | --- | --- |
| beam 1 (greedy) | 7.41 | 7.52 | 46 / 200 |
| **beam 5** (current default) | **8.00** | **8.20** | **44 / 200** |
| beam 5, repetition penalty 1.3 | — | **8.78** | **34 / 200** |
| beam 5, `no_repeat_ngram_size=2` | — | 8.36 | 23 / 200 |

Mean latency ≈ 390 ms per sentence at beam 5, which is inside the MT share of the 2.0 s
turnaround budget.

## What this says

1. **The leg is real.** EN→VI produces coherent Vietnamese at 8.20 corpus BLEU, in the
   same band as VI→EN (9.84 mean sentence BLEU on the FLEURS MT manifest, ADR-032). Neither
   direction is strong; both are usable, and neither is a blocker before 13 November.
2. **Greedy is not usable in this direction either.** At beam 1 the model loops on the
   Vietnamese target side (`năm năm năm năm`, `chí chí chí`) — the same pathology the class
   already guards against on the VI→EN side. Beam 5 is the floor, not an optimisation.
3. **Repetition is systematic, not incidental**: 22% of beam-5 outputs contain a repeated
   trigram. This is a demo-quality risk — a listener hears the stutter even when BLEU is
   acceptable.
4. **The guard is cheap to improve.** Raising `repetition_penalty` to 1.3 lifts corpus BLEU
   to 8.78 *and* cuts repetitions to 17%. That is a decode-setting change, not a model
   change. Treat the two tuned rows as indicative rather than as a result: they were
   selected on the same 200 items they are scored on.

## Recommended setting for the app

`beam_size=5, repetition_penalty=1.3, no_repeat_ngram_size=3` for both directions until a
proper sweep says otherwise. This is a recommendation, not yet a decision — it changes the
decode configuration in `kavi-tuning.yaml`, so it should be ratified on the device in phase
2 (register-length clauses, real audio) rather than on FLEURS.

## The register-length check, which qualifies everything above

Those BLEU figures are measured on FLEURS sentences (median ~22 English words in, ~40
Vietnamese tokens out). The app decodes **clauses**, so the same model was run on short
expressions (3–11 words) — and there it degrades structurally.

On the device (`ct2_bench`, beam 5, 256-token cap) five unique short clauses produced about
**200 output tokens each, 4 of 5 hitting the cap**. On the host the same inputs hit their
`guarded_length` cap of 48 tokens — that is, the host was **truncating** the degeneration
rather than avoiding it, which is what made the aggregate BLEU look tolerable: the harness
caps output length in proportion to source length (`bench/decode.py`).

Raising the guard does not cure it. With the cap raised to 256 to expose the behaviour:

| guard | sample behaviour |
| --- | --- |
| 1.1 / ngram 3 (current) | 72, 158, 256 tokens on three inputs; `du du du khách khách du khách` |
| 1.5 / ngram 2 | 38, 38, 256 tokens; loops reduced on two, one unchanged |
| 2.0 / ngram 2 | 37, 27, **256 tokens** |

At least one input (`fellow wrestlers also paid tribute to luna`) degenerates to the cap at
every setting, so this is a **capability floor** of the model on short input containing
proper nouns, not a decode-tuning problem.

### What this means for the submission

1. **Do not read 8.20 as the register figure.** At clause length the direction is usable but
   unreliable, and the honest statement is "MT EN→VI degenerates on a material fraction of
   short inputs".
2. **Keep the length guard.** `guarded_length` in `ct2_stream_jni.cpp` bounds the damage: the
   user hears a truncated sentence instead of 256 tokens of looped Vietnamese. It is a
   mitigation, not a fix.
3. **Raise the penalty to 1.5 / ngram 2 in the app** and have phase 2 add a repetition
   detector that retries once at 2.0. Cheap, in-app, and testable on register audio.
4. **M2M-100 becomes the EN→VI successor for the finale**, on this evidence rather than on
   FLEURS BLEU: it covers en→vi, it is the only candidate covering VI↔ZH and VI↔KO, and the
   register-level quality floor is now the binding constraint on this leg.

This does not change the submission plan: EN→VI ships, guarded and disclosed. It does change
what we claim about it.

## Reproduce

```bash
make models                                   # fetches vi→en and en→vi, both pinned
uv run python -c "from bench.data_prep import build_mt_manifest; build_mt_manifest('eval_data/mt_en_vi_eval_manifest.json', n_items=200, direction='en->vi')"
uv run python -m bench.run --manifest eval_data/mt_en_vi_eval_manifest.json \
    --out bench-results/mt-en-vi-b5 --candidate opus-mt-en-vi-ct2-cpu \
    --config-override '{"beam_size": 5}'
make verify-assets                            # 6/6 expected
```
