# Can we train or fine-tune anything on free compute?

Question: the Phase-3 deadline (13/11/2026) is the prototype round, and field testing and the final follow it. Is there a training path worth taking with a few verified Kaggle accounts, Google Colab, and — as a stretch — some rented GPU hours?

**Answer: yes, but only one of the four candidates is worth the window. MT fine-tuning is cheap, licence-clean, and lands on criterion A (45%). TTS distillation is the biggest prize and the wrong project for these 19 days.**

## 1. What the compute actually is

| | Kaggle (free) | Colab (free) |
|---|---|---|
| GPU | 2× T4 16 GB (no NVLink, so 16 GB per model — the second card buys throughput, not size) | 1× T4 16 GB, model not guaranteed |
| Quota | **30 GPU-h/week**, live counter, 20 TPU-h/week (v3-8) | **no published quota**, demand-throttled |
| Session cap | 12 h (9 h TPU) | 12 h nominal, often far less |
| Storage | 20 GB working, 200 GB datasets | ~15 GB ephemeral, Drive for persistence |
| Reality | P100 retired 15/09/2026, so T4×2 is the only free accelerator; demand has spiked since, with accelerators greyed out at peak | unpredictable; worst tool for planned multi-day work |

Practical reading: plan for **20-28 usable T4-hours per account per week**, write checkpoint/resume code, and expect a single 30-hour job to be split across three commits.

Two cautions:

- **T4 is Turing**: fp16 fine with a GradScaler, **no bf16, no FlashAttention-2**. Memory bandwidth (~320 GB/s) is the real constraint, not FLOPs.
- **Multiple accounts need care.** One verified account per *team member* is legitimate. Creating accounts to bypass the quota is what Kaggle's anti-abuse filters exist to catch, and a disqualified team gains nothing. Budget on per-person accounts and treat extra accounts as unavailable.

## 2. The candidates

| project | params | realistic T4-hours | what it buys | licence | verdict |
|---|---|---|---|---|---|
| **MT fine-tune** (Opus-MT vi-en) | 77 M | **5-20** | domain accuracy on the actual scenario; **reliable punctuation** for chunk cutting | Opus-MT CC-BY-4.0, derivatives fine | **do this** |
| ASR fine-tune (Zipformer VI) | ~30 M | 20-60 | WER in factory noise — the field test environment has fans and foot traffic | model Apache-2.0; data licences vary | stretch |
| TTS few-step distillation | 20-160 M | 100-200 | removes the ~620 ms TTS fixed cost | Supertonic OpenRAIL-M; training code not ours | **not now** |
| VITS from scratch (VI) | ~30 M | 100-300 | a licence-clean 1-NFE Vietnamese TTS | data licences vary | not now |

### The correction that changes the TTS story

**VITS is already single-step.** It is a VAE + GAN with a HiFi-GAN decoder: one forward pass, no diffusion, nothing to distil. The earlier framing — "distil a flow model down to 2-3 steps" — only applies to Supertonic, whose vector estimator is flow-matching and therefore pays per step.

So for Vietnamese TTS the choice is not "distil Supertonic" but:

- **VITS/Piper fine-tune** — needs a working Vietnamese phonemiser, which is exactly where the Piper path is already blocked (missing `voices/vi` in the shipped espeak-ng-data);
- **flow-matching fine-tune then distil** — 100-200 T4-h, needs Supertonic's training code, and the literature is blunt that distillation cannot exceed its teacher, so a mediocre teacher makes the whole exercise pointless.

Either way it is a multi-week project with a licence question at the end, competing against a submission that still has no end-to-end measurement.

## 3. Why MT fine-tuning is the one worth doing

It attacks the two measured weaknesses with the smallest budget:

**a) Domain accuracy.** Criterion A is 45%, and the scenario is fixed and known: factory, construction, logistics. Opus-MT is a general model. A few thousand domain sentence pairs move it toward the vocabulary the judges will actually hear.

**b) Punctuation, which is a latency lever in disguise.** The MT/TTS overlap reads target commas to decide where the first spoken chunk ends. Measured on five items, MT emitted clause marks in a pattern of **0, 39, 0, 1, 6** — sparse and inconsistent, and the 39 was a degenerate loop. Because punctuation is unreliable, the cut policy falls back to a word cap, which is what makes cadence coarse. **A model that punctuates reliably turns a latency heuristic into a quality feature**, and costs a fine-tune rather than an architecture.

Feasibility: 77 M params, mixed fp16 on a T4, a few thousand pairs — this is an hours-scale job, not a days-scale one, and it fits comfortably inside one account's weekly quota with room for a second experiment.

### Data

| source | what it gives | licence |
|---|---|---|
| Organizer sample set (100 pairs) | domain-matched, tiny — held out for **evaluation**, not training | supplied |
| IWSLT15 / PhoMT / MTet / OPUS | general VI-EN volume | per corpus; mostly permissive, verify each |
| Curated scenario sentences | the factory/logistics vocabulary the scenario needs | ours |

The honest risk: **100 domain sentences is an evaluation set, not training data.** If we fine-tune on a few thousand general pairs plus a few hundred curated domain pairs, the measurable claim is "domain terms improved on a held-out domain set", which is defensible. Claiming a BLEU jump on 100 sentences would not be.

## 4. What I would not do, and why

- **Distil Supertonic** — no training code we control, 100-200 T4-h, licence review, and the teacher is 5-step quality we have already judged "acceptable, slightly flat". The upside is real (~300-400 ms) but it cannot land before 13/11, and it would compete with the submission for attention.
- **Train TTS from scratch** — the search is unambiguous: from scratch on a T4 with 5-20 h of Vietnamese is a guaranteed failure. The advice is to start from a pretrained flow model or use VITS.
- **Train an end-of-turn detector** — we established that the cheap version does not work (`endpointing-gate.md`); the real version needs labelled dialogue we do not have.
- **Chase TPU quota** — the TTS/MT stack here is PyTorch/ONNX and the TPU path costs more porting time than the quota is worth.

## 5. A 19-day shape that does not threaten the submission

| priority | work | hours | lands criterion |
|---|---|---|---|
| 1 | end-to-end measurement + 30-min scenario + installable package | 0 GPU | B, C, D (55%) |
| 2 | MT fine-tune: punctuation + domain, evaluated on held-out domain set | 10-20 GPU-h | A (45%) |
| 3 | ASR noise fine-tune, only if 1 and 2 are done | 20-40 GPU-h | A |

Training runs in the background of submission work precisely because it is small. If the fine-tune does not beat the baseline on a held-out domain set, it does not ship — and the baseline is still a working prototype.

## 6. Things that would change this answer

- If the qualifying build is comfortably ahead, the TTS distillation path becomes the right post-13/11 project for field testing and the final: 100-200 T4-h is roughly 6-10 weeks of free quota, or ~$35-70 at spot prices as a stretch.
- If the scenario turns out to be mostly short counter-style exchanges, domain MT tuning matters less than latency and the budget should go to #57's UX items instead.
- If a Vietnamese Piper voice becomes available with a working phonemiser, the TTS question simplifies to a fine-tune rather than a distillation research project.

## 7. The arithmetic with three accounts

Three members, one verified account each: **3 accounts**. Each GPU session gets **2× T4**, so three concurrent sessions is **6 T4s** — which is what "6× T4" means in practice, spread across accounts rather than within one job. No single job can borrow another account's quota.

Today is 25/10/2026; the deadline is 13/11/2026, i.e. **19 days ≈ 2.7 weeks**.

One number decides the whole budget, and it must be read off the live counter rather than assumed: **does an hour on a 2×T4 session consume one quota-hour or two?** Kaggle documents "30 GPU hours per week" without settling it, and the counter in the notebook UI is authoritative. Both readings:

| reading | per account/week | all three, per week | available before 13/11 |
|---|---|---|---|
| quota counts session-hours | 30 session-h (= 60 T4-h) | 180 T4-h | **~488 T4-h** |
| quota counts T4-hours | 30 T4-h | 90 T4-h | **~244 T4-h** |

So the planning figure is **244-488 T4-hours before the deadline**, against these requirements:

| project | T4-hours | share of the conservative budget | wall-clock at 90-180 T4-h/week |
|---|---|---|---|
| MT fine-tune (+ punctuation, domain) | 20 | 4-8% | 1-2 days |
| ASR noise fine-tune | 60 | 12-25% | 2-5 days |
| VITS (character-level) from scratch | 100-300 | 20-120% | 6-12 days |
| Supertonic flow distillation | 100-200 | 20-82% | 6-12 days |

**The conclusion is not "we are short of GPU".** At the conservative reading there is enough compute to run the MT fine-tune *and* one multi-week training project before the deadline. The scarce resource is engineering attention: 19 days, of which the submission build needs roughly 12. The GPU-hours are the easy part, which is exactly why they can run asynchronously.

A rented stretch changes this only at the margin: at $0.2-0.5 per T4-hour, 200 T4-hours is **$40-100**. Renting removes the quota ceiling; it does not create engineering days.

### Kaggle mechanics that make asynchronous runs work

- **Use "Save & Run All" (commit) runs, not interactive sessions** — they run headless for up to 12 h without the interactive idle timeout, which is what a background job needs.
- **Chunk every job to fit 12 h** and write checkpoints that resume; a 30-hour fine-tune is three commits, not one.
- **Use Kaggle Datasets as the checkpoint bus** (versioned, immutable, 200 GB per dataset) and Drive as the second copy, so either account can resume someone else's run.
- **Enable internet** in notebook settings when a run needs to pull HF datasets; it is off by default for some run modes.
- **Keep both T4s busy** with `torchrun --nproc_per_node=2` (data parallel). A single-GPU job on a 2×T4 session wastes half the quota, and no NVLink means 16 GB is still the per-model ceiling.

## 8. Revised sequencing

The MT fine-tune is the only training that must land before the deadline. Everything else is asynchronous and scored against the *next* milestone, not this one.

| lane | account | work | ends | ships before 13/11 |
|---|---|---|---|---|
| human | — | end-to-end measurement, 30-min scenario, installable package | continuous | yes, mandatory |
| 1 | A | MT fine-tune: punctuation + domain, held-out domain eval, int8 CT2 export, device re-measure | day 5-7 | **yes** |
| 2 | B | ASR noise fine-tune, evaluated on noisy VI clips | day 10-14 | only if it wins clearly |
| 3 | C | VITS feasibility: train 1 h, listen, then scale to 10-100 h | day 3 gate | no |

Lane 3 deserves explanation, because it is the most interesting project available and the cheapest one to start.

**Character-level VITS sidesteps the licence problem entirely.** The Piper path is blocked because its frontend needs espeak-ng (GPL-3.0, ADR-031) and the shipped espeak data lacks a `voices/vi` entry. But Vietnamese orthography is close to phonemic — the tone is written as a diacritic on the vowel — so a character-level model receives most of the information a phoneme sequence would carry. Training a VITS on characters needs **no espeak, no Piper, and no GPL dependency**, and VITS is **1-NFE**, which is the latency property this project has been chasing.

Its gate is a one-hour experiment: train on 1 h of clean VI speech, listen, and if the character-level output is intelligible, the 10-100 h run is worth the compute. That gate costs ~2 T4-hours and answers the question that has blocked TTS for three rounds.

## 9. Dataset licensing: the corpus we need is the one we cannot use

This is the load-bearing constraint on every training lane, and it is worse than the compute picture. Checked against the rights holders, not Hugging Face YAML tags — mirrors in this space routinely apply `cc-by-4.0` as a default.

| corpus | stated licence | who says so | usable under ADR-029? |
|---|---|---|---|
| **Common Voice vi** (v27.0) | **CC0-1.0** | Mozilla, authoritative | **yes** — but only **7.81 h validated / 22.97 h total**, 419 speakers, read speech |
| **FLEURS vi** | **CC-BY-4.0** | Google, authoritative | **yes** — small (hours), read speech, already local |
| Bud500 (~500 h) | Apache-2.0 | VietAI's own repo | probably — but the card also says "research purposes only", and the audio is compiled from public sources VietAI cannot license |
| VLSP-2020 100 h | *none published* | nobody — the `cc-by-4.0` tag is an unofficial mirror's | **no** — BigScience logged licensing as "unclear"; the release announcement has no licence clause |
| InfoRe №1 (~25 h, single female) | *none published* | nobody — mirrors disagree (CC-BY / NC-SA / NC), and one mirror commit is literally titled "less restrictive license" | **no** — a mirror cannot relax a licence it never held |
| VIVOS (15 h) | **CC BY-NC-SA 4.0** | AILAB-VNUHCM, Zenodo DOI | **no** — non-commercial *and* share-alike |
| Emilia | CC BY-NC 4.0 | Meta | **no** — but see §10, it is the training data behind ZipVoice's released checkpoints |

**The consequence is blunt.** Every Vietnamese TTS paper trains on VLSP-100h, InfoRe-25h or VIVOS — all blocked. What remains licence-clean is **~8 validated hours of Common Voice** (419 speakers, varied microphones, read speech) plus a few hours of FLEURS. That is a fine ASR adaptation corpus and a thin, multi-speaker, mixed-quality TTS corpus.

So the honest position for any TTS lane: **the constraint is data, not GPU.** A licence-clean single-voice Vietnamese TTS cannot be trained to commercial quality on what is legally available, and that is the same wall ADR-029 hit with NLLB, MMS-TTS and Piper.

## 10. Correction: ZipVoice was skimmed, and it should not have been

Earlier revisions of this page dismissed ZipVoice on the strength of one sentence — that `hynt/ZipVoice-Vietnamese-2500h` is CC-BY-NC-SA-4.0 and therefore blocked. That conflated three separable things, and the conflation hid the best available option:

| layer | licence | verdict |
|---|---|---|
| `hynt/ZipVoice-Vietnamese-2500h` (pre-trained VI checkpoint) | CC-BY-NC-SA-4.0 | blocked |
| **k2-fsa/ZipVoice base** — code *and* weights | **Apache-2.0** | usable |
| fine-tuning pipeline (`egs/zipvoice/run_finetune.sh`, `run_custom.sh`) | Apache-2.0 | usable |

The pre-trained *Vietnamese checkpoint* being NC says nothing about whether the *architecture* can be fine-tuned on licence-clean data. It can.

What this changes, concretely:

1. **Few-step comes free.** ZipVoice-Distill ships as released weights at **~4-8 Euler steps with classifier-free guidance folded into the student**, 123M params. The 100-200 T4-hour distillation project described in §2 is not needed — it has already been done upstream. That was the single biggest cost in the TTS lane, and it is not real.
2. **The runtime is already in the app.** sherpa-onnx — the 1.13.8 AAR this project vendors — has a ZipVoice path (`OfflineTtsModelConfig.zipvoice`) and a prebuilt distill-int8 bundle. Unlike VieNeu (ADR-028: estimated 1.5-2 days for a bespoke C++/JNI bridge), there is no new native code.
3. **The path has been walked publicly.** A Vietnamese ZipVoice fine-tune exists (`dinhphuongvn/text_to_speech_Vietnamese`) with the `run_finetune.sh` and `prepare_dataset.py` scripts, so the recipe is not speculative.

Two real caveats, neither fatal:

- **The tokenizer drags espeak back in.** The default `EmiliaTokenizer` uses `piper_phonemize` + espeak-ng for English, and `EspeakTokenizer` covers "any other language" via `--lang vi` — which is GPL-3.0, i.e. ADR-031 again. The mitigation is to fine-tune with a **character-level tokenizer and our own `tokens.txt`**: Vietnamese orthography is close to phonemic and tone is a diacritic, so characters carry most of what phonemes would. Whether sherpa-onnx's ZipVoice path tolerates a model without `espeak-ng-data` is the one thing the gate experiment must check.
- **Apache-2.0 weights, NC training data.** The released checkpoints are trained on **Emilia (CC-BY-NC)**. The Apache-2.0 grant covers the weights as the authors' assertion; it cannot relicense Emilia. The LibriTTS-trained checkpoints use CC-BY-4.0 data but are English-only, so cross-lingual transfer to VI is less certain. **This is the question ADR-029 has to rule on**, and it should be asked before, not after, a fine-tune lands.

### Revised TTS ladder

| option | few-step | runtime cost | licence path | data |
|---|---|---|---|---|
| **ZipVoice-Distill fine-tune** | yes, from released weights | none — already shipped | Apache-2.0 base, Emilia question to resolve | ~8 h CV + FLEURS |
| character-level VITS | yes (1-NFE) | ONNX export, new path | clean throughout | ~8 h CV + FLEURS |
| Supertonic distillation | would be | existing | OpenRAIL-M, no training code | n/a |
| Piper VITS | yes | existing AAR | GPL espeak frontend | blocked |

**The gate experiment is now ~2 GPU-days instead of a research project:** fine-tune ZipVoice-Distill on the CC0 Common Voice Vietnamese slice for ~10k iterations, then compare 4 steps against Supertonic's 5 steps on the same listening criteria we already use. If Common Voice's varied microphones cap the quality, that is learned in two days rather than three weeks — and it is the honest test of whether "few-step plus licence-clean" is reachable for Vietnamese at all.
