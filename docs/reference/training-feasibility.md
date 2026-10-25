# Can we train or fine-tune anything on free compute?

Question: the Phase-3 deadline (13/11/2026) is the prototype round, and field testing and the
final follow it. Is there a training path worth taking with a few verified Kaggle accounts, Google
Colab, and — as a stretch — some rented GPU hours?

**Answer: yes, but only one of the four candidates is worth the window. MT fine-tuning is cheap,
licence-clean, and lands on criterion A (45%). TTS distillation is the biggest prize and the wrong
project for these 19 days.**

## 1. What the compute actually is

| | Kaggle (free) | Colab (free) |
|---|---|---|
| GPU | 2× T4 16 GB (no NVLink, so 16 GB per model — the second card buys throughput, not size) | 1× T4 16 GB, model not guaranteed |
| Quota | **30 GPU-h/week**, live counter, 20 TPU-h/week (v3-8) | **no published quota**, demand-throttled |
| Session cap | 12 h (9 h TPU) | 12 h nominal, often far less |
| Storage | 20 GB working, 200 GB datasets | ~15 GB ephemeral, Drive for persistence |
| Reality | P100 retired 15/09/2026, so T4×2 is the only free accelerator; demand has spiked since, with accelerators greyed out at peak | unpredictable; worst tool for planned multi-day work |

Practical reading: plan for **20-28 usable T4-hours per account per week**, write checkpoint/resume
code, and expect a single 30-hour job to be split across three commits.

Two cautions:

- **T4 is Turing**: fp16 fine with a GradScaler, **no bf16, no FlashAttention-2**. Memory bandwidth
  (~320 GB/s) is the real constraint, not FLOPs.
- **Multiple accounts need care.** One verified account per *team member* is legitimate. Creating
  accounts to bypass the quota is what Kaggle's anti-abuse filters exist to catch, and a
  disqualified team gains nothing. Budget on per-person accounts and treat extra accounts as
  unavailable.

## 2. The candidates

| project | params | realistic T4-hours | what it buys | licence | verdict |
|---|---|---|---|---|---|
| **MT fine-tune** (Opus-MT vi-en) | 77 M | **5-20** | domain accuracy on the actual scenario; **reliable punctuation** for chunk cutting | Opus-MT CC-BY-4.0, derivatives fine | **do this** |
| ASR fine-tune (Zipformer VI) | ~30 M | 20-60 | WER in factory noise — the field test environment has fans and foot traffic | model Apache-2.0; data licences vary | stretch |
| TTS few-step distillation | 20-160 M | 100-200 | removes the ~620 ms TTS fixed cost | Supertonic OpenRAIL-M; training code not ours | **not now** |
| VITS from scratch (VI) | ~30 M | 100-300 | a licence-clean 1-NFE Vietnamese TTS | data licences vary | not now |

### The correction that changes the TTS story

**VITS is already single-step.** It is a VAE + GAN with a HiFi-GAN decoder: one forward pass, no
diffusion, nothing to distil. The earlier framing — "distil a flow model down to 2-3 steps" — only
applies to Supertonic, whose vector estimator is flow-matching and therefore pays per step.

So for Vietnamese TTS the choice is not "distil Supertonic" but:

- **VITS/Piper fine-tune** — needs a working Vietnamese phonemiser, which is exactly where the
  Piper path is already blocked (missing `voices/vi` in the shipped espeak-ng-data);
- **flow-matching fine-tune then distil** — 100-200 T4-h, needs Supertonic's training code, and the
  literature is blunt that distillation cannot exceed its teacher, so a mediocre teacher makes the
  whole exercise pointless.

Either way it is a multi-week project with a licence question at the end, competing against a
submission that still has no end-to-end measurement.

## 3. Why MT fine-tuning is the one worth doing

It attacks the two measured weaknesses with the smallest budget:

**a) Domain accuracy.** Criterion A is 45%, and the scenario is fixed and known: factory,
construction, logistics. Opus-MT is a general model. A few thousand domain sentence pairs move it
toward the vocabulary the judges will actually hear.

**b) Punctuation, which is a latency lever in disguise.** The MT/TTS overlap reads target commas to
decide where the first spoken chunk ends. Measured on five items, MT emitted clause marks in a
pattern of **0, 39, 0, 1, 6** — sparse and inconsistent, and the 39 was a degenerate loop. Because
punctuation is unreliable, the cut policy falls back to a word cap, which is what makes cadence
coarse. **A model that punctuates reliably turns a latency heuristic into a quality feature**, and
costs a fine-tune rather than an architecture.

Feasibility: 77 M params, mixed fp16 on a T4, a few thousand pairs — this is an hours-scale job,
not a days-scale one, and it fits comfortably inside one account's weekly quota with room for a
second experiment.

### Data

| source | what it gives | licence |
|---|---|---|
| Organizer sample set (100 pairs) | domain-matched, tiny — held out for **evaluation**, not training | supplied |
| IWSLT15 / PhoMT / MTet / OPUS | general VI-EN volume | per corpus; mostly permissive, verify each |
| Curated scenario sentences | the factory/logistics vocabulary the scenario needs | ours |

The honest risk: **100 domain sentences is an evaluation set, not training data.** If we fine-tune on
a few thousand general pairs plus a few hundred curated domain pairs, the measurable claim is
"domain terms improved on a held-out domain set", which is defensible. Claiming a BLEU jump on 100
sentences would not be.

## 4. What I would not do, and why

- **Distil Supertonic** — no training code we control, 100-200 T4-h, licence review, and the teacher
  is 5-step quality we have already judged "acceptable, slightly flat". The upside is real (~300-400
  ms) but it cannot land before 13/11, and it would compete with the submission for attention.
- **Train TTS from scratch** — the search is unambiguous: from scratch on a T4 with 5-20 h of
  Vietnamese is a guaranteed failure. The advice is to start from a pretrained flow model or use VITS.
- **Train an end-of-turn detector** — we established that the cheap version does not work
  (`endpointing-gate.md`); the real version needs labelled dialogue we do not have.
- **Chase TPU quota** — the TTS/MT stack here is PyTorch/ONNX and the TPU path costs more porting
  time than the quota is worth.

## 5. A 19-day shape that does not threaten the submission

| priority | work | hours | lands criterion |
|---|---|---|---|
| 1 | end-to-end measurement + 30-min scenario + installable package | 0 GPU | B, C, D (55%) |
| 2 | MT fine-tune: punctuation + domain, evaluated on held-out domain set | 10-20 GPU-h | A (45%) |
| 3 | ASR noise fine-tune, only if 1 and 2 are done | 20-40 GPU-h | A |

Training runs in the background of submission work precisely because it is small. If the fine-tune
does not beat the baseline on a held-out domain set, it does not ship — and the baseline is still a
working prototype.

## 6. Things that would change this answer

- If the qualifying build is comfortably ahead, the TTS distillation path becomes the right
  post-13/11 project for field testing and the final: 100-200 T4-h is roughly 6-10 weeks of free
  quota, or ~$35-70 at spot prices as a stretch.
- If the scenario turns out to be mostly short counter-style exchanges, domain MT tuning matters
  less than latency and the budget should go to #57's UX items instead.
- If a Vietnamese Piper voice becomes available with a working phonemiser, the TTS question
  simplifies to a fine-tune rather than a distillation research project.
