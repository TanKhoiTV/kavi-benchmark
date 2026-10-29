# Code-switching language ID: what the literature and shipping products do

Date 2026-10-29. Companion to [`lid-code-switching.md`](lid-code-switching.md) and [`lid-share-sweep.md`](lid-share-sweep.md), which are our own measurements. This page is **prior art**: what other language pairs have established, and how products that advertise automatic detection actually behave.

It exists because our measured failure — `argmax(mean per-token log-prob)` picking English on 25%-Vietnamese audio — turns out to be a named, studied problem with a standard cure, and because the fix we were about to hand-roll has forty years of prior art in speaker and language recognition.

Labels follow the blueprint's convention: **[PRIOR ART]** is published and cited, **[OURS]** is measured here, **[APPLICATION]** is what we do with it.

## 1. Our task is language diarization, not language identification

**[PRIOR ART]** The literature separates three tasks, and conflating them is a common source of wrong expectations:

| task | question | output | usual metric |
| --- | --- | --- | --- |
| utterance-level LID | which language is this utterance? | one label per utterance | accuracy, macro-F1, EER |
| sub-utterance LID | which language at each short span? | frame/segment labels (~200 ms) | frame accuracy, per-language EER |
| **language diarization** | **which language when, within one recording** | time-stamped segments | language DER, JER, per-language EER |

**[APPLICATION]** We have been asking for one label per turn on audio that may contain two languages. That is language diarization squeezed into a binary decision, and the literature is explicit that the squeeze is where it breaks: Mishra & Prasanna (*Issues in Sub-Utterance Level LID*, SPCOM 2022) document the **dominant-language bias** directly, and the field reports it as the main failure of utterance-level LID on code-switched speech — a minority span contributes few tokens but the averaging step is over all of them.

**[OURS]** That is precisely our statistic: one mean over every token. Our sweep is a measurement of the known-weakest decision rule applied to a pair nobody has benchmarked.

## 2. What other pairs establish

**[PRIOR ART]** There is no single head-to-head table across pairs — different corpora, different decision schemes — but the pattern is consistent:

| pair | LID behaviour on code-switched speech | dominant failure |
| --- | --- | --- |
| Spanish–English | strongest of the well-studied pairs | Spanish↔Portuguese confusion; English over-prediction on very short clips |
| Mandarin–English (SEAME) | mid; strongly duration-sensitive | **English embedded in Mandarin is classified as Mandarin**, because the English is Mandarin-accented |
| Hindi–English (MUCS, ASCEND) | mid-to-low | matrix-language dominance on short segments |

Four findings transfer to us whether or not the pair is VI–EN:

1. **The accent asymmetry is the pivot.** Embedding language segments are produced with the matrix language's phonology, so they are acoustically closer to the matrix than to canonical target. EACL 2026 Findings (*Improving LID for Code-Switched Speech: The Pivotal Role of Accented English*) shows this is **data-driven rather than intrinsic**: 50-500 accented-English samples per accent with adapters/LoRA closes much of the Mandarin/Hindi gap. For us the mirror case applies — Vietnamese speech through an English recogniser — and our measurement shows it is the English model that stays *confident* while doing it.
2. **Short inputs collapse across every method.** Reported zero-shot spoken-LID F1 falls by roughly 10-25 points going from 30 s to under 3 s, and the ranking of methods changes with duration: frame-level/CTC and zipformer systems are far flatter across bins than utterance-level pooling or Whisper's single 30 s language token.
3. **No single threshold separates code-switched from monolingual speech** (SLT 2026, multi-label set-prediction formulation). A binary forced choice is structurally incapable of representing "both", so the honest target is a per-turn *routing* decision with a stated error rate, not a classifier boundary.
4. **Synthetic code-switching needs voice conversion.** SAGE-LD (2025) concatenates monolingual audio to simulate code-switching, and reports that without voice conversion to unify speaker identity, models learn **speaker** diarization instead of language diarization — because in real recordings speaker correlates with language.

**[OURS]** Finding 4 is a defect in our own probe: the spliced constructions pair clip *i* from the Vietnamese set with clip *i* from the English set, so the two halves are different speakers, and a system that learned "whose voice is this" would score well on our data for the wrong reason. The confidence comparison uses no speaker model, which limits the damage, but the construction should be read as an upper bound on separability and never as a simulation of real code-switching.

## 3. The cure is a standard one: normalise, then calibrate

**[PRIOR ART]** Comparing scores from two heterogeneous models is the central problem of the speaker- and language-recognition literature, and it has a settled vocabulary that separates three operations people routinely conflate (Brümmer & du Preez 2006; van Leeuwen & Brümmer, Odyssey 2007):

| operation | does what | needs | fixes |
| --- | --- | --- | --- |
| **normalisation** (Z-, T-, ZT-, S-norm) | affine re-scaling per side to remove condition-dependent shifts | an unlabelled cohort | score-scale mismatch *between* systems |
| **calibration** (logistic LR, PAV, temperature) | monotone map to a calibrated log-likelihood ratio | labelled trials | miscalibration *of* a system |
| **fusion** | weighted combination of systems | labelled dev data | missing information |

**S-norm** is the standard recipe for modern embeddings, and it is exactly shaped for our problem:

```text
s_snorm = ½ · [ (s − μ_enrol) / σ_enrol + (s − μ_test) / σ_test ]
```

i.e. normalise each side of the comparison by its own distribution before subtracting. Matejka et al. (Interspeech 2017) report ~30% relative gain for adaptive S-norm in mismatched conditions, and — the detail that matters here — their adaptively selected cohort **matched the language in ~68% of trials and the gender in ~92%**: the normalisation is implicitly clustering by language, which is what a two-language comparison needs.

**[APPLICATION]** Our measured failure is the textbook symptom the literature predicts when a threshold is applied to un-normalised scores from different systems: *"a single decision threshold cannot be applied across models without normalisation."* Our δ was a hand-tuned one-parameter stand-in for the calibration step, and it failed for the documented reason — it is uniform, while the bias is per-condition.

**[PRIOR ART]** One caution worth carrying: normalisation does **not** produce calibrated likelihoods, and can even *worsen* per-trial calibration when the two systems have different intrinsic variances. The defensible modern recipe is per-model normalisation, then a proper calibrator, evaluated with **C_avg / C_mxe and the Cllr split (Cllr_min vs Cllr_cal)** rather than EER — because pooled multi-class EER is miscalibration-blind by construction.

**[APPLICATION]** Our `P(VI | share)` curve is the right *diagnostic*, and the wrong *metric* to publish. If a detection number goes into the submission, it should be per-language recall plus a calibration-aware figure, computed on held-out labelled audio with the normalisation and calibration parameters fitted on a disjoint set.

**[OURS] Tested, and it does not rescue the comparison.** [`lid-statistic.md`](lid-statistic.md) runs five statistics — mean, median, top-k, z-normalised and t-normalised — over the same token arrays at n=100. Z-normalisation is the best of them at every point above 40% share and the only one with a perfect clean anchor, but at a 25% Vietnamese share every statistic still says Vietnamese 2-3% of the time. The prediction failed for a reason worth recording: the comparison is not *miscalibrated*, it is measuring which model is less damaged, and no monotone transform of two means recovers a decision the arithmetic never encoded. The lesson generalises — a normalisation cure is worth testing before it is worth adopting.

## 4. What shipping products actually do

This is the part that changes our design, because the best hands-free products mostly **avoid** acoustic language ID.

| product | how direction is decided | honest limitation |
| --- | --- | --- |
| **Google Translate, conversation mode** | closed set of the **two selected languages**; forced choice from ASR + LID scoring, with pause-based turn boundaries and turn-history expectations | "Detect language" is **disabled for voice**; no confidence shown and no per-utterance correction; a misdetection produces language A → A, heard as an echo; short utterances, close languages and code-switching are the documented weak spots |
| **Timekettle (flagship mode)** | **which microphone is live** — directional gating locks to each wearer's own mouth, so the direction follows the person, not the language. Real LID only appears in the newer engine-selector generation | auto mode is recommended for calm environments; they push users to tap-to-talk in crowds |
| **Pocketalk S2 (auto two-way)** | genuine **utterance-level LID** on the two preselected languages; still press-to-talk on one shared device | their own documentation warns accuracy drops when the two languages are mixed in one utterance |
| **RTranslator (open source)** | per-result confidence when both hypotheses are plausible; text LID only for 2-6 words, returns "und" at 7+ | confidence is the load-bearing signal, not classification |

**[APPLICATION]** Three design lessons, all of which cost nothing to adopt:

1. **Closed set of two, always.** Every shipping system constrains detection to the pair the user selected. Nobody runs open-set LID for this. We already do the same, and should say so rather than implying general language detection.
2. **The best hands-free experience uses non-acoustic evidence.** Timekettle's flagship avoids LID entirely by knowing whose microphone is live. Our analogue is the **turn history** already in the plan, and it should be treated as a first-class signal rather than a tie-breaker.
3. **Everyone discloses the mixed-language weakness.** Google cannot correct a misdetection; Pocketalk documents the degradation. Our ≤3% monolingual / disclosed-mixed framing is the same posture, and the demo script should say it out loud rather than let a judge discover it.

## 5. So what do we do

Ordered by evidence strength, not novelty:

1. **~~Replace the statistic~~ — done, and it does not work.** Five candidates at n=100 leave a 25% Vietnamese share at 2-3%; see [`lid-statistic.md`](lid-statistic.md). The statistic was not the defect. What survives from that run is z-normalisation as the *best* statistic (keep it: 4-7 points at 70% share, perfect clean anchor) and the finding that top-k confidence selection destroys the signal.
2. **Calibrate, do not hand-tune.** Fit the operating point on held-out audio with a one-parameter logistic map (the honest version of δ), and report per-language recall; keep `P(VI | share)` as the diagnostic plot.
3. **Make turn history explicit.** Both the literature and the commercial systems treat speaker turn as signal. In a two-party exchange it is the only evidence that survives code-switching.
4. **Report at the right granularity.** Per-turn routing accuracy with a stated mixed-speech class, not a single "LID accuracy".
5. **The productive target is time, not arithmetic.** Decide per clause rather than per turn — the pipeline already cuts clauses for MT and TTS, so a per-clause decision sees mostly one language. The sweep's own construction is the argument: it spends most of its range on a contiguous foreign block spliced across two speakers, which is not how either realistic case behaves.

**[APPLICATION]** A caveat on the tempting shortcut: for VI–EN both languages use the Latin script, so text-side LID is *weaker* here than for Mandarin–English — Vietnamese single-syllable function words collide with English tokens, and ASR frequently strips the diacritics that would otherwise be a giveaway. Published token-level LID F-scores for less-resourced pairs sit around 80-85, and are worse on single-word islands. Text plausibility is a tie-breaker, not the mechanism.

## Sources

Verified from the retrieved material; numbers attributed to abstracts or secondary summaries are marked in place rather than presented as primary.

- Liu et al., *End-to-End Language Diarization for Bilingual Code-Switching Speech*, Interspeech 2021 (XSA-E2E; WSTCSMC 2020 dev 81.20% accuracy).
- Mishra & Prasanna, *Issues in Sub-Utterance Level LID in a Code-Switched Bilingual Scenario*, SPCOM 2022; and *Implicit Self-Supervised Language Representation for Spoken Language Diarization*, IEEE/ACM TASLP 2024.
- *SAGE-LD: Scalable and Generalizable End-to-End Language Diarization via Simulated Data Augmentation*, arXiv:2510.00582 (voice conversion to decouple speaker from language).
- DISPLACE 2023 (Baghel et al.) and DISPLACE 2024 (Kalluri et al.), Interspeech — multi-speaker joint speaker/language diarization.
- MERLion Code-Switching Challenge 2023; Wang et al., Interspeech 2024.
- *Improving Language Identification for Code-Switched Speech: The Pivotal Role of Accented English*, Findings of EACL 2026.
- *Code-Switching Spoken Language Identification as Multi-Label Set Prediction*, SLT 2026 (no single threshold separates CS from monolingual).
- Nguyen & Bryant, *CanVEC — the Canberra Vietnamese-English Code-switching Natural Speech Corpus*, LREC 2020.
- ViMedCSS, LREC 2026 (arXiv:2602.12911); VietPrism, arXiv:2609.30005.
- TSPC, *A Two-Stage Phoneme-Centric Architecture for Code-Switching Vietnamese-English Speech Recognition*, arXiv:2509.05983 (CS WER 19.9%).
- Brümmer & du Preez, *Application-independent evaluation of speaker detection*, Computer Speech & Language 2006 (Cllr, PAV, calibration/discrimination split).
- van Leeuwen & Brümmer, *On calibration of language recognition scores*, Odyssey 2007; Brümmer, multiclass calibration and the LRE'07 heterogeneous-system result.
- Matejka, Novotný, Plchot, Burget, Sánchez, Černocký, *Analysis of Score Normalization in Multilingual Speaker Recognition*, Interspeech 2017 (adaptive S-norm, cohort language match).
- Google Translate help, *Translate a bilingual conversation* ("detected between the 2 languages selected"; spoken "Detect language" unavailable); Timekettle technical pages (directional voice capture, engine selector); Pocketalk S2 manual (auto two-way LID; mixed-language caveat); RTranslator 3.0.0-beta2 source.
