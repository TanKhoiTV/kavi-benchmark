# Text-gated endpointing: does the transcript know the sentence is finished?

`pause-distribution.md` showed the silence timeout is boxed in — lowering it truncates, raising it breaks the latency gate — and the speech-dialogue literature's answer is to stop treating silence as evidence of completion and let the transcript decide (kavi-android#56). This page tests whether that evidence exists at all, before anyone builds a model or a device pipeline around it.

**Verdict: at this scale, it does not. A text-only gate cannot recover the 200 ms.**

## Method

Every Vietnamese sentence in the organizer corpus (100 sentences, mean 8.16 words) is cut at every word boundary. Each cut is one sample:

- **truncated prefix** (716 samples) — the gate must say *incomplete*; saying *complete* truncates the turn;
- **full sentence** (100 samples) — the gate must say *complete*; saying *incomplete* only waits longer.

816 samples in total. 5-fold cross-validation **holds out whole sentences**, so a gate cannot memorise the sentence it is scored on.

The two gates are computable from a hypothesis alone — no reference, no knowledge of the total length, nothing the device would not have:

- **final-particle rule:** complete iff the last token is a Vietnamese sentence-final marker (không, chưa, gì, nào, đâu, chứ, ạ, nhé, nhá, đi, rồi, vậy, thế, hả, sao);
- **naive Bayes** over token unigrams, the last two tokens, and a length bucket, Laplace-smoothed.

The asymmetry drives the evaluation. A false *complete* cuts speech; a false *incomplete* only costs latency. So the headline is **recall on true ends at a fixed false-complete budget**, and accuracy is not reported — it would be dominated by the 7:1 class imbalance.

## Results

| gate | AUC | recall @0% FC | @1% | @5% | @10% |
|---|---|---|---|---|---|
| naive Bayes | **0.839** | 0.010 | **0.100** | 0.290 | 0.500 |
| final-particle rule | 0.642 | 0.000 | **0.000** | 0.320 | 0.320 |

At a 1% truncation budget — the strictest setting that does not make things worse than the current 3-in-100 rate — the better gate recognises **10% of true sentence ends**. The other 90% would wait longer, which is the latency the timeout was supposed to save.

### The practical variant is worse than it looks

The obvious cheap design is a hybrid: fire at 300 ms when the partial hypothesis ends in a final marker, otherwise wait for 500 ms. Measured on the same corpus:

| quantity | value |
|---|---|
| fast path fires at the **true** end | 32 of 100 utterances |
| fast path would fire **early**, on a prefix | 23 of 100 utterances |
| average saving if applied everywhere | 64 ms |
| truncation exposure introduced | **23%** |

**23% truncation exposure to buy 64 ms** — against the 3% that the current 500 ms already costs. The hybrid is unusable, and the reason is visible in the sentence inventory: `không`, the most common final particle (18 of 100 sentences end with it), also occurs mid-sentence as the negation, so the rule fires on prefixes constantly.

## Why the text alone is not enough here

- **It is context-free.** The literature's detectors see dialogue history — LiveKit's model reads the last four turns. A bare sentence prefix has none, so "Anh giải thích lại chậm hơn" is genuinely ambiguous: it reads as a complete clause and as the first half of a question.
- **Prefixes are often syntactically complete.** Vietnamese is head-initial with little obligatory marking at the end, so the last-token signal is weak (rule AUC 0.642, barely above chance).
- **The data is tiny.** 100 sentences cannot support a learned detector; the systems that work use a fine-tuned 135M model on thousands of turns.
- **The task here is harder than theirs.** They classify *turns* in a dialogue; this classifies *prefixes of a single sentence*, where the only available evidence is linguistic.

## A correction to the severity framing

Earlier write-ups in this repository and on the issue describe endpointing as *losing* speech. That is wrong, and it matters for how much this is worth fixing.

Endpointing **splits the turn**; it does not discard audio. The remainder is captured as a new utterance and translated on its own. The real costs are therefore:

- the fragment is translated **without its antecedent clause** ("nhưng người giao đang đứng nhầm ở đầu đường" without the sentence it qualifies), which is a quality cost, not a loss;
- the second part pays the full VAD + ASR + MT + TTS chain again, so the listener hears a pause and a restart.

That is a real defect at 3% of utterances, and it is *less* severe than "content is dropped" — which means the bar for spending latency or engineering effort to recover the 200 ms is higher than the original framing implied.

## Recommendation

1. **Keep `vad_speech_timeout_ms: 500`.** The gate already passes at 1849 ms median with it in place, and no gate variant tested here is safe enough to justify going lower.
2. **Do not ship the final-particle rule** as a fast path: 23% truncation exposure for 64 ms.
3. If the 200 ms is ever wanted back, the evidence says it needs one of:
   - **dialogue context and a trained detector** (the literature's architecture — requires data this project does not have, plus a VI-capable model);
   - **the transducer's own end evidence** — a decoder end-of-sentence token or a hypothesis-timestamp gap — which is a different signal from the text-only one tested here and remains untested;
   - **accepting the split** and making the restart cheap, since a split is recoverable and a dropped word is not.

## Limitations

- Sentence-prefix classification is a proxy for a real endpointing decision; the device sees ASR errors and partial hypotheses, neither of which appear here.
- Only the organizer corpus (100 sentences, read speech, single-sentence utterances) was used.
- The naive Bayes gate is deliberately small so the study measures *available information*, not what a well-trained model could do. A transformer would do better — that is the point: it would also need the data and the latency budget we do not have.

## Reproduction

```bash
python -m bench.endpointing_gate --sentences eval_data/vi_sentences.txt \
    --out bench-results/endpointing-gate.json
```

`eval_data/vi_sentences.txt` is the Vietnamese column of the organizer spreadsheet, one sentence per line (not committed: `eval_data` is gitignored).
