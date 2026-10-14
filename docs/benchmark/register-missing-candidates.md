# Plan: Register Missing Benchmark Candidates (M2M-100, Moonshine, Zipformer)

> **Status:** **IMPLEMENTED** — all phases executed 2026-08-06, carried onto
> `feat/benchmark` and open as PR #124. §9 records the DoD and §10 the
> deviations found while executing.
> **Context (owner-confirmed):** the beam-sweep work on `feat/beam-sweep-asr` /
> `feat/beam-sweep-mt` was **exploratory** — it determined the optimal beam width
> per model. That phase is done. On `feat/benchmark` we are now running the
> **actual benchmark**: all models against the real eval manifests (VIVOS / VSS /
> FLEURS slices with the SNR sweep — clean + steady/impulsive @ 15/10/5/0 dB),
> scored on byte-identical inputs.
> **Goal of this change:** make all models *runnable through the standard harness*
> so the actual benchmark matrix can be executed. Not to reproduce sweep tooling.
> **Owner decision needed:** see §8 "Open decisions".

---

## 1. Goal

The benchmark harness (`bench/`) currently registers 7 candidates but can run only
3 (Whisper Small, Opus-MT vi→en, Piper EN). Three model families are on disk with
prior results, but have **no candidate adapter in the active tree**:

| Model family | On-disk models | Stage | Prior results (beam-sweep phase) |
| --- | --- | --- | --- |
| **M2M-100 418M** | `models/m2m100-418m-ct2-int8/` (CT2 int8, ~490 MB) | MT | `bench-results/fluers/m2m100-vi-en-ct2-cpu/` (beam 1/2/4/5/8) |
| **Moonshine Tiny** | `models/moonshine-tiny-ct2/` (CT2 — experiment only; runtime uses HF) | ASR | `bench-results/fluers/moonshine-tiny-{vi,en}-hf-cpu/` (beam sweeps) |
| **Zipformer** | `models/sherpa-onnx-zipformer-vi-30M-int8/` + `models/sherpa-onnx-zipformer-en/` | ASR | `bench-results/fluers/zipformer-{vi-30m,en}-sherpa-onnx-cpu/` (beam sweeps) |

**Target benchmark matrix** (what this change enables):

| Stage | Candidate ids (after this change) |
| --- | --- |
| ASR | `whisper-small-faster-whisper-cpu` · `moonshine-tiny-vi-hf-cpu` · `moonshine-tiny-en-hf-cpu` · `zipformer-vi-30m-sherpa-onnx-cpu` · `zipformer-en-sherpa-onnx-cpu` |
| MT | `opus-mt-vi-en-ct2-cpu` · `m2m100-vi-en-ct2-cpu` |
| TTS | `piper-en-lessac-cpu` |

Each ASR/MT candidate must run the **same per-stage manifest** (with `snr` /
`noise_type` fields from `data_prep.py`'s SNR recipe) via `--candidate <id>` so all
models are scored on byte-identical inputs — the fairness requirement from
`docs/benchmarking-plan.md` §3.5.

---

## 2. Current state (verified 2026-08-06)

### 2.1 Registered candidates (`bench/registry.py`)

| candidate_id | Stage | Status |
| --- | --- | --- |
| `whisper-small-faster-whisper-cpu` | ASR | Runnable (weights fetched from HF at setup) |
| `opus-mt-vi-en-ct2-cpu` | MT | Runnable |
| `piper-en-lessac-cpu` | TTS | Runnable |
| `qnn-whisper-small-htp-v73` | ASR | Stub (`NotImplementedError`) |
| `qnn-opus-mt-vi-en-htp-v73` | MT | Stub |
| `qnn-piper-en-htp-v73` | TTS | Stub |
| `rtranslator-2.1.5` | RTranslator | Read-only adapter for captured on-device outputs |

`bench/candidates/` has exactly one file per registered candidate. There is **no**
`m2m_mt.py`, `moonshine_asr.py`, or `zipformer_asr.py` in the tree.

### 2.2 Harness mechanics (relevant to this work)

- `Candidate.run()` wraps `_infer()` with timing + peak-RAM and guarantees a
  `StageResult` even on failure (`bench/adapters.py`).
- `run_manifest()` caches one candidate instance per `candidate_id` so weights load
  once per run (`bench/run.py`).
- `default_candidate_id_for_stage(stage)` returns the **first** registered id whose
  `stage` matches — used only when `EvalItem.candidate_id` is `None`.
- Config is injected per item via `EvalItem.config` (e.g. `{"beam_size": 4}`),
  merged with registry defaults in `build_candidate()`.
- Scorer (`bench/scorer.py`): WER/CER (jiwer) for ASR, BLEU (sacrebleu) for MT,
  RTF when audio duration is known. `EvalItem.snr` / `noise_type` flow through
  unmodified — aggregation per SNR condition is a reporting step (see §7.4).

### 2.3 Known defect in `bench/run.py` (blocker for the actual benchmark)

Commit `fba5a2b` ("drop no-op candidate_filter pass") accidentally changed the
filter into an `if/elif` chain:

```python
cid = item.candidate_id
if cid is None:
    cid = default_candidate_id_for_stage(item.stage)
cid_label = cid
if cid_label is None:
    cid_label = "unknown"
elif cid != candidate_filter:      # <-- when candidate_filter is None, cid != None is
    continue                        #     always True → EVERY item is skipped
```

**Running without `--candidate` currently skips every item**, and `--candidate X`
only keeps items whose *resolved default* equals X — so it cannot run a non-default
model (e.g. moonshine) against a manifest at all. The actual benchmark depends on
`--candidate` to fan each model over the same manifest. Branch
`feat/beam-sweep-asr` contains the fix (commit `6898497`): `--candidate` **overrides**
`item.candidate_id`; items are skipped only when no candidate resolves.

### 2.4 Dependencies

- `pyproject.toml` already declares `faster-whisper`, `ctranslate2`, `transformers`,
  `torch`, `torchaudio`, `sentencepiece`, `soundfile`, `numpy`, `jiwer`, `sacrebleu`,
  `piper-tts`, `datasets`, `optimum[onnx]`, etc.
- **`sherpa-onnx` is installed in the venv (1.13.4) but NOT declared** in
  `pyproject.toml` / `uv.lock` (0 matches in `uv.lock`). Must be added.
- M2M-100 uses `ctranslate2` + HF `AutoTokenizer`; Moonshine uses HF `transformers`
  + `torch`; Zipformer uses `sherpa-onnx` + `soundfile` + `numpy`. All present in
  the venv today.

### 2.5 Models on disk

- `models/m2m100-418m-ct2-int8/` — CT2 int8; **untracked** (local only).
- `models/moonshine-tiny-ct2/ctranslate2/tiny/` — CT2 format (model.bin, tokenizer,
  vocab). **Moonshine has no CT2 runtime support** (transformers PR #1808 open since
  Oct 2024) — this dir is an experiment artifact; the candidate runs Moonshine via
  HF transformers (weights fetched at setup).
- `models/sherpa-onnx-zipformer-vi-30M-int8/` — `encoder.int8.onnx` + fp32 decoder +
  int8 joiner + `tokens.txt` + `bpe.model`; **untracked**.
- `models/sherpa-onnx-zipformer-en/` — `encoder-epoch-99-avg-1.int8.onnx` (188 MB),
  int8+fp32 decoder/joiner, `tokens.txt`; tracked via git-lfs.
- `models/qnn/` — README only (planned artifacts, nothing to benchmark yet).

### 2.6 Working tree hygiene

`git status` shows unrelated/uncommitted state to clean before landing:

- `M android` (submodule pointer)
- untracked: `bench/scripts/`, `models/m2m100-418m-ct2-int8/`,
  `models/moonshine-tiny-ct2/`, `models/sherpa-onnx-zipformer-en/`,
  `models/sherpa-onnx-zipformer-vi-30M-int8/`, `nul` (junk file, 74 B)

---

## 3. Key discovery: implementations already exist on `feat/beam-sweep-asr`

Branch **`feat/beam-sweep-asr`** (diverged from `feat/benchmark` at `52034bb`)
contains working implementations — they produced the `bench-results/fluers/`
numbers during the exploratory phase:

| Commit | Change | Needed for actual benchmark? |
| --- | --- | --- |
| `a1f5c86` | `bench/candidates/m2m_mt.py` — `m2m100-vi-en-ct2-cpu` | ✅ |
| `c4d31be` | `bench/candidates/moonshine_asr.py` — vi + en classes | ✅ |
| `fa4ca5a` | `bench/candidates/zipformer_asr.py` — vi + en classes | ✅ |
| `2af7d32` | Register Moonshine + Zipformer in `bench/registry.py` | ✅ |
| `6898497` | Fix `--candidate` to override instead of filter (run.py) | ✅ (blocker) |
| `3a2f032` | Case-normalize WER/CER in scorer (Zipformer is ALL CAPS) | ✅ (fair cross-model WER) |
| `1d4a340` | `--config-override` for per-run config injection | ✅ (pin optimal beam per run) |
| `c1976ba` / `8f5d0b0` | Whisper / Opus-MT `beam_size` config-driven refactors | ✅ (uniform config across all candidates) |
| `6956de1` / `f3d2d78` | Language filter / `bench/filter_manifest.py` | ❌ (sweep-era tooling) |
| `d88ad71` / `f4a1d8f` | `bench/aggregate_beam_sweep.py` | ❌ (sweep-era tooling) |
| `4ef5867` + `85d2ab4` | `docs/benchmark/beam-sweep-guide-asr-v2.md` + results | ❌ (historical; keep on branch) |

So this is a **port + reconcile** job for the ✅ items only.

### 3.1 Candidate implementations (what will be ported)

**`bench/candidates/m2m_mt.py` — `M2M100MTCandidate` (`m2m100-vi-en-ct2-cpu`)**
- `ctranslate2.Translator(models/m2m100-418m-ct2-int8, device="cpu", compute_type="int8")`
- HF `AutoTokenizer.from_pretrained("facebook/m2m100_418M", src_lang="vi")` (setup-time fetch)
- **Critical:** `target_prefix=[[en_token]]` forces English output — without it M2M
  produces a random language (BLEU ≈ 0) regardless of beam width.
- Strips the leading language token from hypotheses before decode.
- `config`: `beam_size` (default 4), `repetition_penalty`, `no_repeat_ngram_size`,
  `max_decoding_length`.

**`bench/candidates/moonshine_asr.py` — two classes**
- `MoonshineTinyViCandidate` (`moonshine-tiny-vi-hf-cpu`, `usefulsensors/moonshine-tiny-vi`)
- `MoonshineTinyEnCandidate` (`moonshine-tiny-en-hf-cpu`, `usefulsensors/moonshine-tiny`)
- HF `MoonshineForConditionalGeneration` + `AutoProcessor`; `soundfile` reads,
  mono-mixed, 16 kHz.
- Language guard: errors on items whose `language` ≠ model language (a vi candidate
  never silently transcribes en audio — important for cross-model fairness).
- `config`: `beam_size` → HF `num_beams` (default 1), `early_stopping`,
  `no_repeat_ngram_size=3`, `max_length=448` (raised for 15 s FLEURS clips).

**`bench/candidates/zipformer_asr.py` — base class + two subclasses**
- `ZipformerViCandidate` (`zipformer-vi-30m-sherpa-onnx-cpu`)
- `ZipformerEnCandidate` (`zipformer-en-sherpa-onnx-cpu`)
- `sherpa_onnx.OfflineRecognizer.from_transducer(...)`; int8 encoder preference with
  fp32 fallback globs for encoder/decoder/joiner.
- Beam mapping: `beam=1` → `greedy_search`; `beam≥2` → `modified_beam_search` with
  `max_active_paths=beam` (mathematically distinct algorithms — see docstring).
- Fully offline (models on disk).

---

## 4. Gotchas & risks (must be handled)

1. **`bench/run.py` filter regression** (§2.3) — without the fix, **neither** a
   plain manifest run nor `--candidate <non-default>` works. This is the #1 blocker
   for the actual benchmark.
2. **The sweep branch renamed the Whisper candidate** to
   `whisper-small-multilang-ct2-cpu` (still faster-whisper under the hood) and
   tweaked Opus-MT. **Do NOT port those renames** — keep `whisper-small-faster-whisper-cpu`
   / `opus-mt-vi-en-ct2-cpu` so existing manifests and results stay valid (D6).
3. **`sherpa-onnx` undeclared** in `pyproject.toml` / `uv.lock` — a fresh
   environment cannot run Zipformer.
4. **Zipformer outputs ALL CAPS** → case-sensitive jiwer WER is misleading. Port
   `3a2f032` (lowercase ref+hyp before WER/CER). **Consequence:** the archived
   `bench-results/fluers/zipformer-*` WER numbers predate this fix and are
   pessimistic — the actual benchmark re-runs them anyway, so this is fine, but
   don't compare old zipformer WER against new numbers without noting it.
5. **Setup-time HF downloads** for M2M tokenizer + Moonshine weights. Same pattern
   as faster-whisper today; must be documented (D4).
6. **Registry default ambiguity**: after registration, ASR has 5 candidates, MT 2.
   `default_candidate_id_for_stage` returns first dict match → **ordering matters**.
   Keep Whisper first (ASR) and Opus-MT first (MT) so `candidate_id: null` items
   behave as before (D2).
7. **`--candidate` overrides all items** — running `--candidate moonshine-tiny-vi-hf-cpu`
   against a full (ASR+MT+TTS) manifest writes error rows for MT/TTS items. The
   actual benchmark must use **per-stage manifests** (already the pattern:
   `eval_data/asr_subset_beam_vi.json`, `mt_subset_beam.json`, `vivos_vi_test_manifest.json`,
   `vss_vi_test_manifest.json`, …). If desired, add a `--stage` guard to `--candidate`
   (see §7.3) so a full manifest can be reused safely.
8. **Models are local-only / untracked** (except sherpa-en via git-lfs). Fine for
   local benchmark runs; CI tests stay model-free (existing convention).
9. **Working tree is dirty** (submodule pointer, untracked files, `nul` junk) —
   clean before branching; don't accidentally commit `models/` weights.

---

## 5. Implementation plan (phases)

### Phase 0 — Prep
- Remove junk (`nul`); stash or commit unrelated working-tree changes.
- Branch off `feat/benchmark`: `feat/bench/register-missing-candidates`.
- Baseline: `uv run pytest` must be green before changes.

### Phase 1 — Port candidate files (3 files, with reconciliation)
Copy from `feat/beam-sweep-asr`, then adjust:

1. **`bench/candidates/zipformer_asr.py`**
   - `_model_dir_attr` relative strings → absolute paths from
     `Path(__file__).resolve().parents[2]` (matches `_ROOT` convention in
     `opusmt_mt.py` / `piper_tts.py`) so candidates work regardless of CWD.
   - Keep int8→fp32 fallback globs and the greedy-vs-modified-beam mapping.
   - Keep per-language guards.
2. **`bench/candidates/moonshine_asr.py`** — port the two-class version (matches
   the fluers result ids). Optionally anglicize the Vietnamese error strings
   (cosmetic). Keep module-level `import torch` (declared dep; tests stay
   model-free — no weights load at import time).
3. **`bench/candidates/m2m_mt.py`** — port as-is.

Also port the **`beam_size` config-driven refactors** for Whisper
(`c1976ba`, **id unchanged**) and Opus-MT (`8f5d0b0`) so **every** candidate reads
`config["beam_size"]` uniformly — the benchmark injects the sweep-determined
optimal beam per model through one mechanism. Whisper refactor also moves language
selection to per-item (`item.language`) — needed for EN ASR items in the same
manifest.

### Phase 2 — Register in `bench/registry.py`
- Add 5 entries to `REGISTRY`:
  `m2m100-vi-en-ct2-cpu`, `moonshine-tiny-vi-hf-cpu`, `moonshine-tiny-en-hf-cpu`,
  `zipformer-vi-30m-sherpa-onnx-cpu`, `zipformer-en-sherpa-onnx-cpu`.
- **Keep existing dict ordering** (Whisper first for ASR, Opus-MT first for MT) so
  stage defaults are unchanged.
- Set each candidate's **registry default config to the beam width the sweep phase
  found optimal** (e.g. M2M `beam_size=4` matches the fluers dirs; confirm values
  from `bench-results/archive/beam-sweep-*` / fluers reports). `--config-override`
  can still pin per run.
- Update `default_candidate_id_for_stage` docstring (first-match semantics now
  matters with multiple candidates per stage).

### Phase 3 — Fix `bench/run.py`
- Port `6898497` semantics:
  - `--candidate` **overrides** each item's `candidate_id` (not filter).
  - Skip an item only when no candidate resolves (print a notice).
  - Restore correct behavior with no `--candidate` (repairs the `fba5a2b` regression).
- Port `--config-override` (`1d4a340`): merge into `item.config` before
  `build_candidate()` — pins optimal beam per run without editing manifests.

### Phase 4 — Dependencies
- Add `sherpa-onnx>=1.13.4` to `pyproject.toml`; `uv lock && uv sync`; commit
  `uv.lock`.

### Phase 5 — Scorer normalization (`bench/scorer.py`)
- Port `3a2f032`: lowercase `ref` + `hyp` before `jiwer.wer/cer`. Applies uniformly
  to all ASR candidates → fair cross-model WER per SNR condition. Note in changelog.

### Phase 6 — Tests (`tests/test_bench.py`, model-free)
- Registry: 5 new ids present, unique, correct stage mapping.
- Defaults unchanged: ASR default still `whisper-small-faster-whisper-cpu`; MT
  default still `opus-mt-vi-en-ct2-cpu`.
- Pure-logic: Zipformer beam mapping (`beam=1 → greedy_search`), M2M
  `target_prefix` construction — safe because heavy imports are lazy inside `__init__`.
- **Regression for the actual benchmark:** `run_manifest` unit test with a fake
  candidate asserting (a) no-filter runs process every item, (b) `--candidate X`
  overrides item ids, (c) `--config-override` reaches the candidate.

### Phase 7 — Benchmark-run support (narrow; sweep tooling is NOT ported)
- ❌ `bench/filter_manifest.py`, `bench/aggregate_beam_sweep.py`, sweep-guide docs
  stay on the exploratory branches (historical).
- ✅ No `--stage` guard (D7): benchmark runs use per-stage manifests (existing
  pattern — `asr_subset_beam_vi.json`, `mt_subset_beam.json`, …).

### Phase 8 — Docs & changelog
- Update candidate table in `docs/benchmarking-todo.md` (§3 "Models to benchmark"):
  mark M2M/Moonshine/Zipformer registered and runnable.
- Add a short "Benchmark run protocol" note (§7 below) referencing this plan.
- CHANGELOG entry under `[Unreleased]` → `### Features`.

### Phase 9 — Verification (actual-benchmark smoke)
1. `uv run pytest` — green.
2. `uv run python -m bench.run --smoke` — **must produce 3 records** (proves the
   run.py fix; today it produces zero).
3. M2M: `uv run python -m bench.run --manifest eval_data/mt_subset_beam.json
   --candidate m2m100-vi-en-ct2-cpu` — BLEU should land in the same ballpark as
   `bench-results/fluers/m2m100-vi-en-ct2-cpu/beam-4/run_results.json`.
4. Zipformer-vi: `uv run python -m bench.run --manifest
   eval_data/asr_subset_beam_vi.json --candidate zipformer-vi-30m-sherpa-onnx-cpu`
   — WER sane (case-normalized; expected to improve on archived numbers, §4.4).
5. Moonshine-vi: same vi ASR manifest (setup-time HF fetch).
6. Confirm `--config-override '{"beam_size": 4}'` changes results vs registry
   default, and that whisper/opusmt now honor it too.

---

## 6. Files touched (summary)

| File | Action |
| --- | --- |
| `bench/candidates/m2m_mt.py` | **Add** (port) |
| `bench/candidates/moonshine_asr.py` | **Add** (port) |
| `bench/candidates/zipformer_asr.py` | **Add** (port + absolute model paths) |
| `bench/candidates/whisper_asr.py` | **Edit** — `beam_size` config + per-item language (**id unchanged**) |
| `bench/candidates/opusmt_mt.py` | **Edit** — `beam_size` config (**id unchanged**) |
| `bench/registry.py` | **Edit** — register 5 candidates + default beam configs |
| `bench/run.py` | **Edit** — `--candidate` override, no-filter fix, `--config-override`, optional `--stage` |
| `bench/scorer.py` | **Edit** — case normalization (WER/CER) |
| `pyproject.toml` | **Edit** — add `sherpa-onnx>=1.13.4` |
| `uv.lock` | **Edit** (generated) |
| `tests/test_bench.py` | **Edit** — registry + run_manifest regression tests |
| `docs/benchmarking-todo.md` | **Edit** — candidate table |
| `CHANGELOG.md` | **Edit** — feature entry |
| `docs/benchmark/register-missing-candidates.md` | This plan |

**Not ported** (sweep-era, stay on exploratory branches): `bench/filter_manifest.py`,
`bench/aggregate_beam_sweep.py`, `docs/benchmark/beam-sweep-*.md`.

---

## 7. How the actual benchmark will use this (run protocol)

Per stage, one manifest, fanned over every candidate — byte-identical inputs:

```bash
# ASR — same manifest, 5 candidates
for cid in whisper-small-faster-whisper-cpu moonshine-tiny-vi-hf-cpu \
           moonshine-tiny-en-hf-cpu zipformer-vi-30m-sherpa-onnx-cpu \
           zipformer-en-sherpa-onnx-cpu; do
  uv run python -m bench.run \
    --manifest eval_data/<stage-manifest>.json \
    --candidate "$cid" \
    --out bench-results/<dataset>/$cid
done
```

- **SNR fairness:** `data_prep.py` already emits `snr` (None=clean, 15/10/5/0) and
  `noise_type` (`clean`/`steady`/`impulsive`) per item; the candidates and scorer
  pass them through untouched.
- **Beam width:** the sweep phase determined the optimal beam per model. Registry
  default config carries that value; `--config-override` overrides for experiments.
- **Outputs:** `run_results.json` per candidate dir, `--out` scoped per dataset.
- **Aggregation (follow-up, not this change):** WER/BLEU per {model, SNR,
  noise_type} is currently post-processing — a small reporting script would produce
  the per-condition tables the actual benchmark reports need. Out of scope here;
  flag as a follow-up item.

---

## 8. Decisions (owner-approved 2026-08-06)

| # | Decision | Choice | Implication for implementation |
| --- | --- | --- | --- |
| D1 | Port strategy | **File-copy + manual reconcile** | Copy the ✅ files from `feat/beam-sweep-asr`; do not cherry-pick commits; reconcile manually (no Whisper id-rename, no sweep docs). |
| D2 | Stage defaults | **Keep Whisper (ASR) + Opus-MT (MT) as defaults** | Preserve `REGISTRY` dict ordering: Whisper first, Opus-MT before M2M. `candidate_id: null` items behave as before. |
| D3 | Beam refactors | **Port the refactors** | Whisper (`c1976ba`) + Opus-MT (`8f5d0b0`) honor `config["beam_size"]`, ids unchanged — all 6 candidates share one config mechanism. |
| D4 | Setup-time HF fetches | **Accept** | M2M tokenizer + Moonshine weights fetched from HF at setup (same pattern as faster-whisper). No vendoring into `models/`. |
| D5 | Case normalization | **Globalize in the scorer** | `bench/scorer.py` lowercases ref+hyp before jiwer WER/CER for all ASR candidates. |
| D6 | Whisper id | **Keep `whisper-small-faster-whisper-cpu`** | Do not port the `whisper-small-multilang-ct2-cpu` rename. Existing manifests/results stay valid. |
| D7 | Run organization | **Per-stage manifests** | No `--stage` guard. Benchmark runs use stage-scoped manifests (existing pattern: `asr_subset_beam_vi.json`, `mt_subset_beam.json`, …). |

All decisions are resolved; the implementation plan (§5) is final and proceeds as written. Follow-up items remain out of scope: SNR-aggregated reporting (§7.4).

---

## 9. Definition of done

- [x] All 5 new candidate ids resolvable via `build_candidate()` and listed in
      `bench/registry.py`.
- [x] `python -m bench.run --manifest <any>` (no `--candidate`) processes all items
      (regression fixed).
- [x] `--candidate <new id>` runs the candidate over a stage manifest; results
      written under `--out`; no accidental wrong-stage error rows when using
      per-stage manifests (or `--stage` guard).
- [x] `--config-override '{"beam_size": N}'` honored by **all** ASR/MT candidates
      (whisper, opusmt, m2m, moonshine, zipformer).
- [x] `uv run pytest` green; registry + run_manifest regression tests included
      (26 passed as of PR #124).
- [x] `sherpa-onnx` declared in `pyproject.toml`; `uv.lock` updated. *(The
      declaration was made on the pre-rewrite branch and lost when the history
      was rewritten; it was re-added on `feat/benchmark` in PR #124. It is
      pinned `==1.12.40` because 1.13.x needs onnxruntime C API 27, which the
      `qairt` group's 1.17.1 pin cannot load.)*
- [x] Zipformer spot-run WER sane (case-normalized); M2M spot-run BLEU in the same
      ballpark as archived `bench-results/fluers/` numbers.
- [x] CHANGELOG + `docs/benchmarking-todo.md` updated.
- [x] Working tree clean; no accidental commits of `models/` weights; sweep-era
      tooling left on its branches.

## 10. Implementation notes (deviations found while executing)

1. **sherpa-onnx vs onnxruntime conflict.** `sherpa-onnx>=1.13.4` is built against
   onnxruntime C API 27 but the env has ORT 1.17.1 (pinned by the `qairt` group
   for QAIRT SDK 2.31 compat). Fixed by pinning `sherpa-onnx==1.12.40` **and**
   declaring `sherpa-onnx-core==1.12.40` explicitly — the core wheel ships
   `onnxruntime.dll` next to `_sherpa_onnx.pyd`, making the runtime
   self-contained and independent of the env's ORT. Verified 1.12.40 loads the
   Zipformer models.
2. **`uv lock` needs `--index-strategy unsafe-best-match`.** The universal lock
   fails on the py3.14-win split otherwise (`unbabel-comet` ->
   `torchmetrics<0.11` has no cp314 wheel). Pre-existing latent issue, surfaced
   by any full re-resolution; not caused by this change. Note for future dep
   bumps: either pass the flag or narrow `requires-python`.
3. **Windows console encoding.** `print_table` crashed printing Vietnamese
   diacritics on the cp1252 console; added `sys.stdout.reconfigure(errors="replace")`
   in `bench/run.py` `main()` (same pattern the sweep scripts already used).
4. **Sweep-branch `run.py` had a latent bug** (undefined `cid_label` in the
   `except` handler) — fixed to use `cid` when porting.
5. **Verification parity** (all on `eval_data` subsets, vs archived `bench-results/fluers/`):
   - M2M beam 4: mean BLEU **17.19 = 17.19** (exact). This is the `eval_data`
     subset used for the parity check. The figure reported in the benchmark
     results, **17.50**, is the full 347-item FLEURS pool. Both are correct for
     their own manifest — they are not the same number.
   - Zipformer-vi beam 5: mean WER **0.1375 = 0.1375** (exact); beam 1: **0.1460 = 0.1460**
   - Moonshine-vi beam 1: mean WER **0.499 vs 0.524** (same ballpark; small
     drift — case normalization now applied, minor manifest/audio differences)
