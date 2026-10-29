## [Unreleased]

### Bug Fixes

- Resolve pi-lens false positives via pyrightconfig.json
- Resume large FLEURS parquet downloads in download_fleurs
- Build real-speech FLEURS set (decode audio bytes, pair VI->EN)
- Emit EN ASR items and honor per-item language in Whisper
- **bench:** Set ASR reference via reference_text, not transcript_ref
- **bench:** Emit clean ASR condition once instead of per-noise-type
- **bench:** Stop Opus-MT decode loops; scope WER/CER to ASR only
- **bench:** Bound FLEURS audio read to sampled rows (resolve #34)
- **bench:** Select FLEURS audio by id, not position
- **bench:** Drop no-op candidate_filter pass in run.py
- Lint cleanup for eval_denoising.py
- **android:** Remove stale .dlc reference in MainActivity.kt
- **android:** Remove stale .dlc reference in MainActivity.kt
- **bench:** Guard peak_ram_mb against missing resource module on Windows (#80)
- **qnn:** Use repo-root-relative paths in calibration input lists (#85)
- **changelog:** Stop duplicating the PR link in generated entries([#127](https://github.com/TanKhoiTV/kavi-benchmark/pull/127))
- **bench:** Fail with a message when a --dim value is not an integer
- **training:** Authenticate with API tokens and read the real quota
- **training:** Make the dataset upload and the kernel mount actually work
- **training:** Tokenise targets with spm_target, not spm_source
- **docs:** Correct the audit's ADR-022 finding, and record the repairs

### Documentation

- Refresh stale layout/index docs to match current repo
- Verify InfoRe + 25hours_single licenses (stay avoid)
- Sync harness docs with the implemented v0 benchmark harness
- **benchmarking-todo:** Add progress & to-do snapshot at top
- **license:** Resolve QAIRT runtime gate — ADOPT (clean)
- **license:** Record PKLA (signed 2026-07-19) findings
- Fix PKLA expansion and clarify PhoST benchmarking vs redistribution
- Add Phase-4 QNN conversion & comparison plan (#51)
- **adr-003:** Turn determination method into executable plan (#52)
- Expand CONTRIBUTING with OS-agnostic dev setup + OS notes (#55)
- **phase-4:** Fold verified QNN command reference + pitfalls (#57)
- **phase-4:** Correct .dlc claims and elaborate conversion plan
- Point eval_manifest_v1.json references at eval_data/
- Pin Python 3.12 and document sync prereqs
- Point AGENTS.md reference at .pi/AGENTS.md
- Update agent guidance in .pi/AGENTS.md
- Mirror parent contest-info.md and specifications.md
- Add handoff notes to .pi/HANDOFF.md
- Update stale blocker references (QAIRT EULA resolved, FLEURS available)
- Add PLAN.md for Phases 4–7 with cross-checked task breakdown
- Update PLAN.md — mark Batch 1 done, add Batch 2 task breakdown
- Fix stale SDK paths, venv names, and .dlc references
- **adr:** Accept ADR-005, update follow-up status table (#82)
- Consolidate ADR directory, move ADR-005/006 to docs/decisions/ (#83)
- Defer Piper QNN per ADR-005 Decision 3
- **adr:** ADR-007 production inference architecture and service layer (#89)
- ADR-008 — v1 Android Dual Zipformer ASR decision 
- ADR-009 — v1 Android TTS decision (Supertonic Phase 1 → VieNeu-TTS Phase 2) (#93)
- Add ADR-010 — ALL_OPT decoder optimisation analysis (#94)
- Index NDK conversion runbook in docs/README
- Reconcile plans with ADR-007-010; archive superseded docs to reference/ 
- Add ADR-012 — ASR thread tuning strategy (Proposed) (#100)
- Plan Kotlin + C++ implementation for the android app (#97)
- **repo:** Document that the android app is not vendored here
- Correct kavi-android repo structure after untracking
- Correct README for public publication
- Document public setup, pinned assets and Windows
- **adr:** Split multi-decision ADRs into one decision per record
- **adr:** Add the ADR index, withdraw ADR-004, move the comparison doc
- **adr:** Repoint ADR cross-references after the split
- Tag code-fence languages in repo markdown
- Correct stale references and index gaps in docs and ADRs([#109](https://github.com/TanKhoiTV/kavi-benchmark/pull/109))
- Phase-4 conversion does not require a Windows host([#119](https://github.com/TanKhoiTV/kavi-benchmark/pull/119))
- **adr:** Flag ADR-020 beam-width claim as unmeasured([#120](https://github.com/TanKhoiTV/kavi-benchmark/pull/120))
- **agents:** Describe the benchmark track, not the pre-move layout([#149](https://github.com/TanKhoiTV/kavi-benchmark/pull/149))
- Give the tracked docs one shape and guard it([#153](https://github.com/TanKhoiTV/kavi-benchmark/pull/153))
- **adr:** Pin Opus-MT vi<->en as the translation baseline([#155](https://github.com/TanKhoiTV/kavi-benchmark/pull/155))
- **bench:** Rerun the beam sweep on the guarded decode path([#162](https://github.com/TanKhoiTV/kavi-benchmark/pull/162))
- **bench:** Record the MT encoder's HTP profile([#163](https://github.com/TanKhoiTV/kavi-benchmark/pull/163))
- Record the competitive and literature context for the latency design
- Read the review corpus for the UX half of perceived latency
- Record the Phase 3 submission requirements and the gaps they expose
- Research the free-compute training path
- Work the three-account compute arithmetic and lane plan
- Check dataset licences, and correct the ZipVoice dismissal
- Give AI Hub a second look and narrow the dismissal
- Add the handoff snapshot and make the intro honest
- Run the doc checker over the handoff file
- **bench:** Plan the MT fine-tune data against the measured baseline
- **bench:** Count the FLEURS train split at fetch time, not from memory
- **decisions:** Price beam=4 on the device for open parameter #6
- **bench:** Correct the Opus-MT licence and flag the ASR NC-ND terms
- **bench:** Record the Apache-2.0 ASR checkpoint as the shipping one
- Record the shipped ASR checkpoint and its data provenance
- **bench:** Qualify the EN->VI baseline with the register-length failure
- **bench:** EN->VI ships on M2M-100, measured on the register
- **bench:** Record both MT directions through the app bridge
- **bench:** Text conditioning is worth 20 BLEU
- **bench:** VI->EN should move to M2M, measured
- **bench:** The terminal mark is worth 1.5 BLEU
- Normalize line endings and unwrapping in the MT baseline note
- Bring the public status pages up to the 2026-10-29 measurements
- **gates:** Record the shipped models and the measured timings
- **notice:** Attribute the models that actually ship
- Keep the gate arithmetic in the private track
- **bench:** Record COMET in both directions, and index the register page
- **adr:** Supersede the Opus-MT pin, and audit a register that had rotted
- **bench:** The detection mechanism is biased, not just miscalibrated
- **bench:** Prior art for the detection mechanism, and what shipping products do

### Features

- Scaffold host-side v0 benchmark harness (Phases 0-3)
- Build lean eval set + SNR recipe in data_prep (Phase 1)
- **bench:** Support a real noise bank in data_prep
- **qnn:** Add Phase 4 QNN conversion pipeline setup
- Add Phase 4 QNN setup, Phase 5 RTranslator prep, Phase 6 denoising gate
- **qnn:** Add Whisper Small ONNX export script with fixed-shape encoder
- **qnn:** Add Piper surgery script, calibration lists, and QNN adapter stubs
- **qnn:** Whisper decoder patch + Opus-MT encoder conversion (#75)
- Add .kavi.yaml — single source of truth for pinned config values (#98)
- **scripts:** Add PR review-status dashboard, restore root AGENTS.md([#125](https://github.com/TanKhoiTV/kavi-benchmark/pull/125))
- **qnn:** Prepare the Opus-MT w8a16/w8a8 A/B (P0-P2)([#123](https://github.com/TanKhoiTV/kavi-benchmark/pull/123))
- **ci:** Assert the public boundary in make check([#137](https://github.com/TanKhoiTV/kavi-benchmark/pull/137))
- **bench:** Add M2M-100, Moonshine, Zipformer and Hy-MT candidates with v1 benchmark results([#124](https://github.com/TanKhoiTV/kavi-benchmark/pull/124))
- **bench:** Port the VAD noise benchmark onto the post-move layout([#151](https://github.com/TanKhoiTV/kavi-benchmark/pull/151))
- **bench:** Size the decode cap from the input, not the model([#161](https://github.com/TanKhoiTV/kavi-benchmark/pull/161))
- **bench:** Derive AI Hub input specs from the ONNX graph
- **bench:** Justify vad_speech_timeout_ms from measured pause distributions
- **bench:** Test whether a text gate can replace the silence timeout
- **training:** Set up the MT fine-tune lanes across four Kaggle accounts
- **training:** Require punctuated targets and ASR-shaped source
- **training:** Add the pcfix lane that isolates the tokenizer fix
- **training:** Cap target brevity and stop shipping step checkpoints
- **training:** Add the pc2 ladder on the brevity-capped corpus
- **bench:** Pin and measure the EN->VI leg
- **bench:** Score the organizer register with BLEU, chrF++ and COMET
- **bench:** Measure what code-switching does to the confidence signal
- **bench:** The VI-share sweep, and the bias it exposes

### Miscellaneous

- Add kavi-android submodule (prototype/android)
- Pin kavi-android to runtime-bundled commit
- Pin kavi-android to README consistency fix
- Bump kavi-android submodule to CONTRIBUTING (#1)
- Skip git-cliff changelog-bot commits
- Clarify Makefile targets and drop unused PY var
- Move eval_manifest_v1.json into eval_data/
- Ignore *cache/ directories
- Exclude caches and dot-dirs from pyright
- Declare pyarrow, ctranslate2, and numpy as direct deps
- Normalize dev dependency array to uv canonical single-line form
- Move AGENTS.md into .pi/
- Add handoff prompt to .pi/prompts
- Untrack .pi agent config directory
- Add Phase 4 deps + qairt-env.sh
- Consolidate stray paths, fix SDK path, update .gitignore
- Migrate qairt-converters venv into repo as dependency group
- Update android submodule (stale .dlc refs removed)
- Update android submodule (doc fixes)
- **gitignore:** Ignore generated QNN artifacts, keep README (#81)
- **repo:** Ignore the unvendored android path
- **ci:** Drop the submodule opt-out from checkout
- Stop tracking third-party artifacts and personal docs
- Normalize line endings and make local checks deterministic
- **env:** Make the QAIRT helper portable and add a Windows one
- **env:** Default ANDROID_NDK_ROOT to the installer's SDK root([#118](https://github.com/TanKhoiTV/kavi-benchmark/pull/118))
- Add CODEOWNERS so PRs auto-request a reviewer([#130](https://github.com/TanKhoiTV/kavi-benchmark/pull/130))
- **scripts:** Sync the boundary check with the umbrella copy([#138](https://github.com/TanKhoiTV/kavi-benchmark/pull/138))
- Name this repo kavi-benchmark in its own docs and config([#148](https://github.com/TanKhoiTV/kavi-benchmark/pull/148))
- **codeowners:** Describe the repo under its current name([#150](https://github.com/TanKhoiTV/kavi-benchmark/pull/150))
- **ci:** Mint the changelog token with the App's client ID([#154](https://github.com/TanKhoiTV/kavi-benchmark/pull/154))
- **data:** Move the organizer corpus into the repo, and stop reading ~/Downloads
- **deps:** Declare the AI Hub tooling as an opt-in qairt group
- **assets:** Pin the MT that ships, and document the checkpoint we cannot fetch
- **bench:** Label the register run's host decode time as a diagnostic

### Refactoring

- **bench:** Make manifest candidate-agnostic + cheap FLEURS reads
- **bench:** Cache candidate instances per cid in run_manifest
- **repo:** Untrack the private android submodule
- **env:** Resolve SDK/NDK paths in one platform-aware module([#136](https://github.com/TanKhoiTV/kavi-benchmark/pull/136))
- **config:** Split .kavi.yaml into published gates and private tuning([#146](https://github.com/TanKhoiTV/kavi-benchmark/pull/146))
- Trim the benchmark repo to the host-side track([#147](https://github.com/TanKhoiTV/kavi-benchmark/pull/147))

### Styling

- **training:** Apply ruff format to the launch and data-prep scripts

### Testing

- Add bench harness tests + pytest config

### Build

- Bump kavi-android to AGP 8.10.0 / Gradle 8.11.1 (API 36)
- **deps:** Bump actions/checkout from 7.0.0 to 7.0.1 (#87)
- **deps:** Bump astral-sh/setup-uv from 8.3.2 to 9.0.0 (#86)
- Pin third-party assets and add fetch/verify tooling
- **deps:** Bump orhun/git-cliff-action from 4.8.0 to 4.9.0
- **deps:** Bump astral-sh/setup-uv from 9.0.0 to 10.2.0
- **deps:** Bump orhun/git-cliff-action from 4.9.0 to 4.9.1([#117](https://github.com/TanKhoiTV/kavi-benchmark/pull/117))
- **deps:** Bump astral-sh/setup-uv from 10.2.0 to 10.3.0([#144](https://github.com/TanKhoiTV/kavi-benchmark/pull/144))
- **deps:** Bump actions/setup-node from 7.0.0 to 7.1.0([#135](https://github.com/TanKhoiTV/kavi-benchmark/pull/135))
- **deps:** Bump actions/create-github-app-token from 2.2.2 to 3.3.0([#145](https://github.com/TanKhoiTV/kavi-benchmark/pull/145))

### Ci

- Guard changelog footer for repos without tags
- Declare latest Node.js LTS in workflows
- Bump actions to Node 24 runtime (checkout/setup-node @v7)
- Use commit.remote instead of deprecated commit.github
- Bump astral-sh/setup-uv to v8 (Node 24 runtime)
- Pin astral-sh/setup-uv to v8.3.2 (resolvable tag)
- Pin GitHub Action tags to full commit SHAs
- Stop recursing into the private kavi-android submodule
- **changelog:** Serialise changelog runs and rebase before pushing
- Push CHANGELOG.md with a GitHub App token([#126](https://github.com/TanKhoiTV/kavi-benchmark/pull/126))
- Scope the lint job to read-only and stop persisting credentials([#121](https://github.com/TanKhoiTV/kavi-benchmark/pull/121))
- Scope changelog permissions to the job that needs write([#122](https://github.com/TanKhoiTV/kavi-benchmark/pull/122))
[unreleased]: https://github.com/TanKhoiTV/kavi-benchmark/compare/v0.1.0...HEAD

