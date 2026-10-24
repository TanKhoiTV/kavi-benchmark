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

### Refactoring

- **bench:** Make manifest candidate-agnostic + cheap FLEURS reads
- **bench:** Cache candidate instances per cid in run_manifest
- **repo:** Untrack the private android submodule
- **env:** Resolve SDK/NDK paths in one platform-aware module([#136](https://github.com/TanKhoiTV/kavi-benchmark/pull/136))
- **config:** Split .kavi.yaml into published gates and private tuning([#146](https://github.com/TanKhoiTV/kavi-benchmark/pull/146))
- Trim the benchmark repo to the host-side track([#147](https://github.com/TanKhoiTV/kavi-benchmark/pull/147))

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

