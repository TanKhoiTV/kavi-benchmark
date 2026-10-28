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

## M2M-100 en→vi on device: the direction is salvageable

Measured 2026-10-27 with `bench/mt/ct2_bench` on the Meizu 21 Note (phone free memory 6.8 GB;
constraints deliberately loosened for this run — turnaround ≤ 5 s, peak memory ≤ 6 GB). Inputs are
the **organizers' 100 gold pairs**, so this is the contest register and the BLEU is against their
Vietnamese references. `target_prefix=__vi__`, source tokens carry `__en__`, 4 intra-threads.

| run | p50 | p95 | max | peak RSS | corpus BLEU vs register gold |
| --- | --- | --- | --- | --- | --- |
| M2M-100, beam 1 | **577 ms** | 1061 ms | 1790 ms | **646 MB** | **21.51** |
| M2M-100, beam 5 | 1050 ms | 1940 ms | 3368 ms | 678 MB | 20.87 |
| M2M-100, beam 5, 21-word sentences | 2428 ms | 5035 ms | 5035 ms | 698 MB | — |

Quality is not marginal — it is correct translation:

| English | M2M beam 5 | organizers' gold |
| --- | --- | --- |
| Could you explain that again more slowly? | Bạn có thể giải thích điều này một lần nữa chậm hơn không? | Anh giải thích lại chậm hơn được không? |
| Which room are you in so I can check? | Bạn đang ở trong phòng nào để tôi có thể kiểm tra? | Anh đang ở phòng nào để tôi kiểm tra? |
| How long until the drinks are ready? | Bao lâu cho đến khi đồ uống đã sẵn sàng? | Bao lâu thì có đồ uống? |

### What this settles

1. **EN→VI ships on M2M-100, not Opus-MT.** 21.51 BLEU on the register with no degeneration against
   Opus's 8.20 on FLEURS *and* clause-length loops. ADR-032 pinned the direction but its own
   "M2M is the named successor" clause is now the operative decision for this leg.
2. **Beam 1 beats beam 5 here** (21.51 vs 20.87) at half the latency. Unlike VI→EN, this direction
   does not need beam search — measure, do not assume, one direction's tuning does not transfer.
3. **Latency fits the loosened budget and is borderline for the real one.** Register MT p50 577 ms
   (RTF 0.17 against the 3.34 s mean register clip) leaves the 2.0 s gate reachable only with VAD +
   ASR + overlapped TTS: 500 (VAD) + ~250 (ASR) + 577 (MT) + 710–1206 (first TTS chunk) ≈ 2.0–2.5 s.
4. **Memory is affordable but not free**: 646–698 MB against Opus-MT's 211 MB. The full stack
   (ASR ~450 + MT ~690 + TTS ~400) lands near 1.5 GB, which is above ADR-016's 1.16 GB figure and
   far below the 6 GB ceiling this run used.
5. **Long utterances are the latency risk**: 21-word input gives p95 5.0 s, so the clause-level cut
   policy is what keeps MT inside budget, not the model choice.

### Through the app's own bridge (the numbers the app will actually see)

Re-measured inside the APK with `MtDirectionTest`, which uses `Ct2Translator`'s
native bridge rather than the host harness — same 100 register pairs, 4 threads, clause-level
decode with the two-tier cut policy:

| direction | model | p50 | p95 | corpus BLEU vs register gold | peak RSS |
| --- | --- | --- | --- | --- | --- |
| VI→EN | Opus-MT vi→en | **126 ms** | 368 ms | 20.85 | 299 MB |
| EN→VI | M2M-100 418M | **570 ms** | 1029 ms | **21.32** | 805 MB |

Two things this caught that the host harness could not: the forced `__vi__` prefix and the `</s>`
token arrive as decode steps, so without filtering they would be handed to TTS and spoken; and
register sentences are far cheaper than FLEURS ones (VI→EN p50 126 ms here against 507 ms on
22-word FLEURS input), which matters because the earlier budget arithmetic used the FLEURS figure.

## Text conditioning is a stage, not a detail

The first end-to-end run of the app pipeline exposed something no per-stage benchmark showed:
MT degrades badly on **ASR-shaped text**. Measured on the 100 register pairs, VI→EN, greedy:

| input shape | corpus BLEU | mean hypothesis words |
| --- | --- | --- |
| gold text (cased, punctuated) | **20.32** | 11.7 |
| ASR-shaped: UPPERCASE, no punctuation | **0.14** | 88.9 |
| UPPERCASE + terminal period | 0.59 | 25.9 |
| lowercased, no punctuation | 4.97 | 41.7 |
| **lowercased + terminal period** | **11.51** | 15.8 |

Two effects, both large and both cheap to fix:

1. **Casing costs ~20 BLEU.** Uppercase input — which is exactly what both ASR checkpoints
   emit — drives Marian into 89-word repetition loops. Lowercasing alone recovers most of it.
2. **The terminal stop is worth ~7 BLEU** and halves output length (4.97 → 11.51, 41.7 → 15.8
   words). ASR produces no punctuation, so the pipeline appends a period when the transcript
   does not already end in `.?!`.

`SpeechPipeline` applies both before tokenising. The residual gap (11.51 against 20.32 on clean
text) is what punctuation restoration would buy, and it is now a measured target rather than a
suspicion.

### Integration cost

The app must send the `__en__` source token and `__vi__` target prefix, so `Ct2Translator` needs
target-prefix support — a native-bridge change rather than a config flag. `ct2_bench` already
exercises this path (`argv[7]`), so the device harness is ahead of the app here.

## Reproduce

```bash
make models                                   # fetches vi→en and en→vi, both pinned
uv run python -c "from bench.data_prep import build_mt_manifest; build_mt_manifest('eval_data/mt_en_vi_eval_manifest.json', n_items=200, direction='en->vi')"
uv run python -m bench.run --manifest eval_data/mt_en_vi_eval_manifest.json \
    --out bench-results/mt-en-vi-b5 --candidate opus-mt-en-vi-ct2-cpu \
    --config-override '{"beam_size": 5}'
make verify-assets                            # 6/6 expected
```

---

# VI→EN on ASR-shaped text: M2M beats Opus by 9.8 BLEU

The pipeline feeds MT lowercase, unpunctuated run-ons with an appended stop (above). That is
exactly the input Opus-MT degrades on, so the obvious fix was not more punctuation heuristics
but a different model. 100 register pairs, corpus BLEU:

| model | input | BLEU | mean hyp words | MT p50 on device |
| --- | --- | --- | --- | --- |
| Opus-MT vi→en | clean, cased, punctuated (ceiling) | **26.05** | 9.3 | 126 ms |
| Opus-MT vi→en | ASR-shaped (what ships today) | **13.83** | 13.0 | 162 ms |
| **M2M-100 418M** | ASR-shaped | **23.66** | 7.5 | ~410 ms first clause |
| M2M-100 418M | clean | 28.54 | 7.9 | — |

M2M on degraded input nearly matches Opus on clean input, and produces **42% fewer words**
(7.5 vs 13.0) — fewer words is also less TTS audio on the critical path.

## Punctuation heuristics do not close the gap

| input shaping | BLEU |
| --- | --- |
| period only (current) | 13.83 |
| comma every 3 words | 8.33 |
| comma every 4 words | 10.05 |
| comma every 5 words | 12.30 |
| comma before conjunctions (`và`, `nhưng`, `vì`, `nếu`, …) | **15.03** |
| ceiling with real punctuation | 26.05 |

Mechanical commas make it worse; grammatical guesses buy +1.2. This confirms punctuation
restoration would need a model, and that M2M is the cheaper answer to the same problem.

## What it costs in the pipeline

VI→EN, in-app, VAD-segmented, 5 diffusion steps, 3 MT threads:

| MT model | first audio p50 | p95 | BLEU |
| --- | --- | --- | --- |
| Opus-MT | 1320 ms | 1386 ms | 13.83 |
| **M2M-100** | **1588 ms** | 1593 ms | **23.66** |

**+268 ms for +9.8 BLEU, still 400 ms under the 2.0 s gate.** And because EN→VI already needs
M2M, moving VI→EN to it means **one model for both directions**: one tokenizer, one prefix
scheme, and Opus-MT (211 MB resident) is dropped instead of co-resident with M2M (~750 MB).

Remaining cost is on disk: M2M is 490 MB against Opus's 73 MB.

## The terminal mark a question deserves

ASR output carries no punctuation, so the pipeline appends a terminal mark before MT. Always
appending `.` tells the model that a question is a statement, which is measurable:

| VI->EN, greedy, 100 register pairs | BLEU |
| --- | --- |
| always `.` (what shipped) | 19.92 |
| mark chosen from the final particle (`không`, `chứ`, `nhỉ`, `hả`, `sao`, `chưa`, `gì`, `à`, `ạ`) | **21.39** |
| gold punctuation | 25.55 |

The rule fires on 21/100 utterances where gold has 35/100 questions and is correct on 86/100, so
this is a partial but free recovery of the punctuation gap — the same +1.47 BLEU that a beam
sweep costs 217 ms for. Gold punctuation still leads by 4.2 BLEU, which is the size of the prize
the missing Vietnamese punctuation model is holding.
