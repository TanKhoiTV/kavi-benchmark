# Training — MT fine-tune, four accounts, eight T4 lanes

The one training lane that can land before 13/11, per [`docs/reference/training-feasibility.md`](../docs/reference/training-feasibility.md) §8 and the plan in [`docs/reference/mt-finetune-data-plan.md`](../docs/reference/mt-finetune-data-plan.md).

**What runs where, and why:** training runs on Kaggle, **scoring runs here**. The organizer corpus is not redistributable, so it never leaves this machine; the only thing that crosses the boundary is a checkpoint. That split is not a convenience — it is the public/private rule applied to data.

## What eight T4s actually buys

Four accounts × 2×T4 is **four concurrent sessions, not one eight-way job**: there is no network between accounts. The parallelism therefore has to be *across configurations*. Two ways to spend it:

| spend | effect |
| --- | --- |
| 2×T4 per job (DDP) | each config finishes sooner; fewer configs explored |
| **1 T4 per job, two jobs per account** | **twice as many configs in the same wall clock** ← chosen |

A 77M Marian model gets little from PCIe DDP (Kaggle has no NVLink), and the open questions are *which* configuration wins, not how fast one finishes. Pass `--ddp` if a lane turns out to need the other shape.

## The first decision to make, before planning the week

Kaggle documents "30 GPU hours per week" **without saying whether an hour on a 2×T4 session consumes one quota-hour or two**. The account page counter is the only authority. Both readings for four accounts over the ~2.7 weeks to the deadline:

| reading | per account/week | all four | available before 13/11 |
| --- | --- | --- | --- |
| session-hour | 30 session-h (= 60 T4-h) | 180 T4-h | **~488 T4-h** |
| T4-hour | 30 T4-h | 90 T4-h | **~244 T4-h** |

Run `python -m training.launch probe --account m1 --minutes 15`, note the counter before and after, and record the answer here. At the conservative reading there is still enough for the whole sweep plus a second wave; the scarce resource is attention, not GPU.

## Setup

```bash
# 1. credentials: one verified account per person, four people
cp training/accounts.toml.example training/accounts.toml   # then fill in 4 username/key pairs
python -m training.launch check                            # validates + prints the plan

# 2. the corpus (once; needs network, writes ~2.2 GB under eval_data/, gitignored)
python -m training.prepare_data

# 3. stage + upload code, base model and corpus as a private dataset, per account
python -m training.launch dataset

# 4. push the lanes
python -m training.launch run --wave 1      # lane 1 on each account
python -m training.launch run --wave 2      # lane 2 on each account
python -m training.launch status
```

`--dry-run` works before or after the subcommand and stages everything without calling the API — that is how the generated notebooks are reviewed.

**One UI step the API cannot do:** the notebook must be switched to the **T4 x2** accelerator in the Kaggle UI. `enable_gpu` in `kernel-metadata.json` only asks for "a GPU"; the x2 shape is a notebook setting. Lanes run fine on one T4, they just take longer.

## The lane matrix

Interleaved across accounts (`training/lanes.py`), so one account dying does not destroy a whole comparison — baseline plus a probe on each, not two adjacent configs.

| lane | lr | epochs | other | why |
| --- | --- | --- | --- | --- |
| `mix-lr3e5-e3` | 3e-5 | 3 | — | baseline at the conventional Marian rate |
| `mix-lr3e5-e1` | 3e-5 | 1 | — | is one epoch enough for a domain shift |
| `mix-lr1e5-e3` | 1e-5 | 3 | — | conservative: least forgetting of FLEURS |
| `mix-lr5e5-e3` | 5e-5 | 3 | — | aggressive: fastest domain fit, regression risk |
| `mix-lr3e5-e6` | 3e-5 | 6 | — | more exposure vs overfit |
| `mix-lenp05` | 2e-5 | 3 | `--length-penalty 0.5` | mild brevity pressure |
| `mix-lenp10` | 2e-5 | 3 | `--length-penalty 1.0` | strong brevity pressure |
| `general-lr3e5-e3` | 3e-5 | 3 | `--train train_general.tsv` | **control**: no register slice |

`mix-*` lanes train `train_mix.tsv` (90k pairs, 40% comma, 32% question — matched to the supplied corpus's measured 42% and 35%).

## Scoring a lane locally

```bash
# the pinned baseline first, so every delta has a reference (reproduces 24.95 / 22.13)
python -m training.convert_and_eval --ct2 models/opus-mt-vi-en-ct2 \
    --gold <path to the organizers' xlsx> --beam 1 --out eval_data/train/base_greedy.json

python -m training.launch pull --lane mix-lenp05
python -m training.convert_and_eval \
    --ckpt eval_data/kaggle/output/mix-lenp05/ckpt/final --convert \
    --gold <xlsx> --beam 1 --baseline eval_data/train/base_greedy.json \
    --out eval_data/train/mix-lenp05.json
```

That prints sentence BLEU, corpus BLEU, comma and terminal rates, words per hypothesis against gold, degenerate count, and a **paired bootstrap CI** against the baseline. A delta inside the CI is not a result — at n=100 the interval is ±4 to 7 sentence BLEU, which is why the plan requires a ~400-pair authored held-out set before anything ships.

## Ship gate

From the plan §5. A fine-tune ships only if **all** hold on the build that would ship:

1. domain: beats the greedy baseline with a CI excluding zero;
2. brevity: ≤1.10× gold words (greedy output is currently 1.27×);
3. punctuation: comma ≥40/100, terminal ≥98/100 (never let it drop);
4. device: guarded greedy MT stage p50 no worse than 507 ms.

Matching the beam=4 baseline (28.82 sentence BLEU) *at greedy latency* is the stretch target, not the gate.

## What this corpus is not

There is **no tier C**: the 2,000-5,000 authored service-encounter pairs the plan calls the domain signal do not exist yet, so wave 1 is a **control**. It answers whether register-filtered general data moves the domain number at all. If it does not, that is the evidence that justifies the authoring day; if it moves things a little, tier C is added on top. Tier B (WikiMatrix, CC BY-SA) is excluded pending the share-alike ruling, and the factory slice (tier D) belongs to criterion C's scripted scenario, not to this corpus.

Also deliberately absent: PhoMT (research-only), MTet (NC-SA), IWSLT15 and TED2020 (TED-derived NC-ND), CCMatrix/NLLB (no grant / NC), OpenSubtitles (varies), CCAligned (research-oriented), QED (mixed).

## Licensing, per source

| source | version | licence | how verified |
| --- | --- | --- | --- |
| ParaCrawl-Bonus | v9 | CC0 1.0 | licence string in the OPUS package README |
| Tatoeba | v2026-07-08 | CC BY 2.0 FR | ditto, plus the package `LICENSE` file |
| KDE4 | v2 | GPL (text); the model is not a derivative text | package README points at the source terms |
| Opus-MT vi→en | `c8d2853e` | CC BY 4.0 / Apache-2.0 per ADR-032 | `assets.lock.toml` |
| FLEURS (`vi_vn` train) | — | CC BY 4.0 | not yet included; the *test* split is eval-only |

Attribution for Tatoeba and Wikipedia-derived text must appear in the submission's credits if those rows are used.

## Files

| file | role |
| --- | --- |
| `prepare_data.py` | OPUS → filtered, register-matched, comma/question-balanced TSV |
| `finetune_marian.py` | Kaggle-side trainer: fp16, wall-clock budget, checkpoint/resume, optional DDP |
| `lanes.py` | the eight-lane matrix and the account assignment |
| `launch.py` | per-account dataset upload, kernel push, status, pull, quota probe |
| `convert_and_eval.py` | host-side: HF → CT2 int8, score the held-out pairs, bootstrap CI |
| `accounts.toml.example` | credential template (the real file is gitignored) |
