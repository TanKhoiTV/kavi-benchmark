# MT fine-tune: data plan (VI→EN)

Status: **plan, not yet executed.** Written 2026-10-25 to unblock the only training lane that can land before 13/11 ([`training-feasibility.md`](training-feasibility.md) §8, lane 1). It records the measurements that reshaped the plan, then the three decisions the plan has to close: **corpus mix**, **held-out domain eval set**, **objective**.

The plan as originally framed had a punctuation objective motivated by clause marks measured on five items. That motivation does not survive a 100-item measurement, so this page leads with the measurement rather than the objective it was going to justify.

## 1. What the plan was going to say, and what the evidence says instead

### 1.1 The punctuation deficit does not reproduce

`training-feasibility.md` §3 justifies the punctuation objective with a five-item on-device clause-mark count of **0, 39, 0, 1, 6** — sparse and inconsistent. The 39 in that sample was the degenerate loop that [`decode.py`](../../bench/decode.py) was written to stop, so the sample was measuring a bug, not a habit.

With the guard in place, on the organizer's 100 VI→EN pairs, the pinned `opus-mt-vi-en-ct2` candidate emits commas at **45/100** against a gold rate of **42/100**, and terminal marks at 98/100 against gold 100/100. On 100 unpunctuated FLEURS references it emits commas at **71/100** against a gold rate of 1/100 — it over-punctuates there, it does not under-punctuate anywhere.

| register | gold commas | model commas (greedy) | gold terminal | model terminal |
|---|---|---|---|---|
| organizer domain pairs | 42/100 | **45/100** | 100/100 | 98/100 |
| FLEURS test (sample of 100) | 1/100 | **71/100** | 0/100 | 38/100 |

**Consequence:** punctuation is a constraint to preserve, not a target to raise. A training run optimised to increase comma emission would move the model the wrong way on both registers.

### 1.2 What the punctuation objective was standing in for

The supplied pairs put the first comma **before** the 4-word first-chunk cap on 23/100 items, at or after it on 19/100, and there is no comma at all on 58/100. So reliable punctuation can shorten the first spoken chunk on **at most ~23%** of utterances. That is a cadence lever on 42% and a latency lever on 23% — real, but bounded far below "a latency lever in disguise".

### 1.3 The domain is service encounters, not factory work

The supplied 100 pairs are not factory, construction or logistics vocabulary despite the brief's criterion C persona. They are urban service encounters, roughly: hotel/travel/tourism (~30), retail/shopping (~15), office and events (~15), directions and transport (~12), parcel and warehouse (~10), clinic and pharmacy (~8). Mean length is 8.46 English words (median 8, max 25); Vietnamese mean 8.16 (median 7, max 20). 35/100 are questions.

This contradicts the corpus advice in [`benchmarking-plan.md`](benchmarking-plan.md) §3.3, which proposes authoring "factory/logistics-flavoured sentences" as the domain set. Training on that vocabulary optimises a distribution the supplied evidence says the quality benchmark is not probing. Criterion C still needs factory vocabulary — but for the **scripted 30-minute scenario**, which is a different artefact from the hidden sentence benchmark ([`phase3-submission-requirements.md`](phase3-submission-requirements.md) §Scoring).

### 1.4 The baseline the fine-tune has to beat

Same candidate, same decode guard, 100 held-out pairs:

| config | sentence BLEU | corpus BLEU | commas | words/hyp | vs gold (8.46) |
|---|---|---|---|---|---|
| greedy (**shipped**) | **24.95** | 22.13 | 45/100 | 10.76 | **1.27×** |
| beam=4 (**benchmarked**) | **28.82** | 29.33 | 43/100 | 9.31 | 1.10× |
| m2m100-418m greedy | 27.66 | 28.14 | 39/100 | 7.81 | 0.92× |
| m2m100-418m beam=4 | 29.28 | 29.70 | 40/100 | 7.99 | 0.94× |

Paired bootstrap over items, 10k resamples, 95% percentile CI on the mean difference in sentence BLEU:

| comparison | difference | 95% CI | crosses zero |
|---|---|---|---|
| m2m100 − opus-mt, greedy | +2.71 | [−2.20, +7.54] | **yes** |
| m2m100 − opus-mt, beam=4 | +0.47 | [−4.24, +4.79] | **yes** |
| opus-mt beam=4 − opus-mt greedy | **+3.87** | **[+1.12, +6.78]** | no |

Two consequences, both load-bearing:

- **On this register, M2M-100's advantage is not demonstrable.** ADR-032's overturn trigger is M2M-100 at +78% mean sentence BLEU, which was measured on 347 FLEURS items. On 100 domain pairs the difference is +2.71 greedy and +0.47 at beam=4, both inside the noise band, for 5.8× the parameters and 1102 MB against 399 MB peak RAM. Testing whether that FLEURS margin was a punctuation artefact: it is not — stripping punctuation from both sides moves the FLEURS margin from +104% to +103%. The margin is **register-dependent**, not punctuation-dependent.
- **Beam=4 beats greedy significantly on this register**, +3.87 [+1.12, +6.78]. That is new evidence for open parameter #6, whose quality case was previously settled on FLEURS ([`beam-sweep-mt.md`](beam-sweep-mt.md)); the device turnaround cost remains the binding half.

### 1.5 The measured outlier is verbosity, and it is mostly decode-side

Greedy output runs **1.27× the gold word count**; beam=4 runs 1.10×; M2M-100 runs under 1.0×. So two-thirds of the verbosity excess is a decode setting, not a model defect, and the fine-tune inherits only the remainder.

Verbosity is a latency lever only in a specific sense, and the distinction matters: the gate is EOS→first audio, and the first chunk is a fixed 4 words, so total output length does **not** move first-audio latency. It moves the MT decode (decoder-bound, roughly linear in target tokens) and the total synthesis time for the turn. MT is demoted off the critical path in the shipping configuration, so the honest claim is *turn completion and cadence*, not *the gate*.

## 2. Decision one: corpus mix

### 2.1 The licence wall applies to MT text, not only to speech

`training-feasibility.md` §3 lists IWSLT15, PhoMT, MTet and OPUS as the volume, annotated "mostly permissive, verify each". Verified, the first three are blocked by the same gate the page applies elsewhere in §9 — the page never applied its own licensing method to the MT corpora:

| corpus | pairs (vi–en) | stated terms | ADR-029 |
|---|---|---|---|
| **PhoMT** (VinAI) | 3.02M | research/education only, no redistribution, cite | ❌ blocked |
| **MTet** (VietAI) | 4.2M | contradictory: arXiv `CC BY 4.0` describes the *paper*; the HF dataset card says **CC BY-NC-SA 4.0**; repo silent | ❌ blocked as written |
| **IWSLT15** en–vi | 133K | TED-derived → **CC BY-NC-ND 4.0** plus TED usage policy | ❌ blocked |
| TED2020, News-Commentary, GlobalVoices, JW300 | — | NC / ND / research-only | ❌ blocked |
| CCAligned / MultiCCAligned, CCMatrix, QED, OpenSubtitles | — | no explicit grant, research-oriented, mixed, or "varies per source" | ⚠️ not cleared |

The derived-weights question — whether a model fine-tuned on NC text is itself NC — is legally unsettled, and this project has already deferred exactly that question once, for ZipVoice ([`training-feasibility.md`](training-feasibility.md) §10). The plan does not depend on it: the mix below is clean **before** any ruling, so training can start while ADR-029 is still open.

### 2.2 The mix

| tier | source | licence | role | volume target |
|---|---|---|---|---|
| **A — clean core** | Tatoeba vi–en | CC BY 2.0 FR (attribute) | general VI↔EN volume, colloquial register closest to the domain | ~100K available; sample |
| **A — clean core** | ParaCrawl vi–en (**verify the `vi` column exists in the release used**) | CC0 1.0 | general web volume, no conditions | sample to size |
| **A — clean core** | HPLT / MultiHPLT | CC0 | general volume alternative | sample to size |
| **A — clean core** | free-software localisation: GNOME, KDE4, Ubuntu, Mozilla-I10n | GPL/LGPL/MPL/FDL — copyleft binds the text, commercial use of the model is fine | short UI-style imperatives, the closest public analogue to "please wait a moment" | small, high value |
| **A — clean core** | FLEURS `vi_vn` **train** split | CC BY 4.0 | clean short-sentence volume; **never the test split**, which the published benchmark is scored on | ~3K |
| **B — needs a ruling** | WikiMatrix / Wikipedia | CC BY-SA 4.0 — attribution *and* share-alike; whether SA reaches weights is contested | optional volume if A is too thin | defer |
| **C — authored** | service-encounter pairs written by the team, in the register §1.3 measures | ours | **the domain signal** — the only data that targets the held-out distribution | 2,000–5,000 |
| **D — authored** | factory/construction/logistics pairs | ours | criterion C's scripted scenario only; not expected to move the hidden sentence benchmark | 300–800 |

Two rules that make the mix defensible:

1. **Every source enters with its licence recorded**, as an `assets.lock.toml` entry or a row in this table, with the version string (`Tatoeba v2023-04-12`, `ParaCrawl v9`) — OPUS terms change between versions, and the version is what a compliance review cites.
2. **Tier C carries the domain, tier A carries the general distribution.** A mix that is only tier A fine-tunes toward web/formal register and will not move the held-out number; a mix that is only tier C will forget FLEURS. The budget split that follows is roughly 80% tier A / 20% tier C by pair count, weighted toward tier C per epoch.

## 3. Decision two: held-out domain eval set

### 3.1 What is scorable today

| direction | on disk | scorable |
|---|---|---|
| **VI→EN** | 100 clips, VI transcript **and** an English reference per item | **yes** |
| **EN→VI** | 100 clips, no transcript | **no** |

ADR-032 records that "the organizers' EN→VI sample carries no reference transcript to score against". That is right about the EN audio and wrong by conflation about the VI→EN pairs, which do carry `Tiếng Anh - Final` references. The EN→VI leg is additionally unbuilt, not merely unmeasured: `models/` holds only `opus-mt-vi-en-ct2`, so no `opus-mt-en-vi` weights exist locally to score. Fixing either half is out of scope for this plan and should be tracked separately.

### 3.2 The supplied 100 are held out, and they are too small alone

They are the evaluation set, never training data — `training-feasibility.md` §3 says so, and it is now a hard rule: any authored tier-C pair must be checked against the supplied 100 for near-duplicates before it enters training.

But n=100 cannot resolve the improvement this plan is trying to detect: the bootstrap CIs in §1.4 are ±4–7 sentence BLEU wide. A fine-tune that genuinely gains 2 BLEU would land inside that band and be indistinguishable from noise. **The plan therefore requires a second held-out set of ~400 authored domain pairs**, disjoint from both the supplied 100 and the tier-C training data, split so that the same page can report a CI rather than a point estimate. That is a day of authoring and it is what makes the ship/no-ship decision in §5 meaningful.

### 3.3 Metrics, named

Report **both** BLEU definitions — mean sentence BLEU and corpus BLEU — because they disagree in direction elsewhere in this repo ([`beam-sweep-mt.md`](beam-sweep-mt.md)), and a single unqualified "BLEU" is the kind of number ADR-032 already had to correct. COMET is worth adding only if `Unbabel/wmt22-comet-da` can be run offline; it is not required by the plan.

## 4. Decision three: the objective

Replacing "punctuation objective" with what the measurements support:

| objective | criterion | rationale |
|---|---|---|
| **Domain fit** | beat the greedy baseline (§1.4) on the held-out domain sets, with a bootstrap CI that excludes zero | the only thing the training is for |
| **Brevity** | word ratio ≤ 1.10× gold at greedy decode (from 1.27×) | §1.5; beam=4 already reaches 1.10×, so this asks the fine-tune to hold a setting's benefit at a setting's price |
| **Punctuation preservation** | comma rate ≥ 40/100 and terminal ≥ 98/100 on the held-out set | §1.1; a constraint, and an explicit regression guard because the easy failure is a model that decodes fluently and stops punctuating |
| **General-distribution guard** | FLEURS `vi_vn` test corpus BLEU not below 10.03 greedy beyond noise | the published benchmark must not regress while the domain number rises |

The brevity criterion is the one that needs care in execution: length penalty is a training-side hyper-parameter with a direct quality trade, so it is swept **after** domain fit is established, not trained jointly. If it moves domain BLEU down beyond the CI, brevity is dropped and the finding recorded — beam=4 remains the lever for it.

## 5. Shipping gate

The fine-tune ships only if all four hold, measured on the build that would ship:

1. Domain: beat greedy baseline with a CI excluding zero, **on the ~400-pair authored set**, and consistent in sign on the supplied 100.
2. Brevity: ≤1.10× gold words.
3. Punctuation: comma ≥40/100, terminal ≥98/100.
4. Device: guarded greedy MT stage p50 no worse than the current 507 ms.

Matching the beam=4 baseline (28.82 sentence BLEU) *at greedy latency* is the stretch target, not the gate. If a run fails 1–4 it does not ship and the baseline prototype stands, exactly as `training-feasibility.md` §5 states.

## 6. Verification: what is now executable

Both of these are available today, so neither leg is a projection:

- **On-device.** The target phone enumerates over ADB as `MEIZU_21_Note` / `SM8550` / Android 16, so the int8 CTranslate2 export can be re-measured on the device rather than on the host. The MSYS path-conversion caveat and the harness list are in the private handoff.
- **AI Hub.** `qai_hub` 0.56.0 is importable in this repo's environment. It is a portability and quantization-QA lab, not a latency source ([`aihub-capability-review.md`](aihub-capability-review.md)): run `numerics` on the fine-tuned decoder to prove the int8 conversion did not move outputs, and check offload per chipset.

The verbosity→latency link in §1.5 is also measurable rather than argued: synthesise one held-out item at its greedy hypothesis and at a beam=4-length hypothesis and compare total synthesis time on the device. That converts "brevity is a lever" from inference to measurement, and it costs one harness run.

## 7. Risks, and what would change the answer

- **The tier-C authoring is the real cost, not the GPU.** 2,000–5,000 domain pairs in a specific register is a writing-and-review job; machine-assisted drafting without review imports the authoring model's distribution, which is the thing being fixed.
- **The register inference rests on 100 items and one directory.** If the hidden benchmark is factory-flavoured prose rather than service encounters, tier C is aimed at the wrong distribution and tier D becomes the domain signal. Any additional supplied sample would settle it cheaply, and it is the first thing to ask the organizers for.
- **`ParaCrawl` may not carry Vietnamese** in the release used, which is why the `vi` column is an explicit verification step rather than an assumption.
- **Fine-tuning a 77M Marian model on a few thousand pairs can degrade more than it fixes.** The FLEURS guard in §4 exists to catch exactly that; if it trips, the honest reading is that the domain is worth an authored glossary or a decode-time constraint, not a fine-tune.
- **A larger beam width may simply be the better answer.** If open parameter #6 resolves to beam=4, the fine-tune is competing with a configuration change that already gains +3.87 BLEU on this register for latency rather than authoring effort.

## 8. Reproduction

The organizer corpus is not redistributed; the commands name its shape, not a path on anyone's machine. Scripts that consume it live outside every repository.

```bash
# domain baseline, both decode configs, against the supplied references
python baseline_domain.py --out baseline_domain.json

# register guard: the same model on unpunctuated FLEURS references
python fleurs_punct.py

# paired bootstrap over items (hyps cached; no re-translation on re-run)
python domain_bootstrap.py
```

## References

- [`training-feasibility.md`](training-feasibility.md) — the compute budget, the lane plan this page executes, and §9-10's licensing method applied here to MT text
- [`phase3-submission-requirements.md`](phase3-submission-requirements.md) — scoring pillars, the 30-minute scenario, the measurement-not-estimate rule
- [`benchmarking-plan.md`](benchmarking-plan.md) §3.3 — the earlier corpus advice this page contradicts on register
- [`beam-sweep-mt.md`](beam-sweep-mt.md) — the FLEURS beam sweep, and the two BLEU definitions §3.3 asks for
- [`ADR-032`](../decisions/ADR-032-mt-model-choice.md) — the MT pin, its M2M-100 overturn trigger, and the EN→VI gap §3.1 narrows
- [`../../bench/decode.py`](../../bench/decode.py) — the decode guard whose absence produced the five-item measurement in §1.1
- TanKhoiTV/kavi-benchmark#110 — ADR-032's record
- TanKhoiTV/kavi-benchmark#115 — open parameter #6, given new domain-register evidence in §1.4
