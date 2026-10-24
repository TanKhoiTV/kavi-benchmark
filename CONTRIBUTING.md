# Contributing to Kavi (kavi-prototype)

## Repository role

This is the **implementation** submodule of `aivoice-2026` (the umbrella repo).
It contains the code and internal docs; public-facing documentation belongs in
the parent. Third-party model weights and eval corpora are **not** committed —
they are fetched on demand (see [Models & data](#models--data)).

## Cloning

`kavi-prototype` is a **Git submodule** of the public parent repo
[`aivoice-2026`](https://github.com/TanKhoiTV/aivoice-2026). That means it does
**not** come down with a normal `git clone` of the parent. Both repositories are
public, so no access request is needed.

### 1. Clone the parent with the submodule in one step

```bash
git clone --recurse-submodules git@github.com:TanKhoiTV/aivoice-2026.git
cd aivoice-2026
```

No SSH key? Use HTTPS instead: `https://github.com/TanKhoiTV/aivoice-2026.git`.

### 2. Or clone first, then initialize the submodule

```bash
git clone git@github.com:TanKhoiTV/aivoice-2026.git
cd aivoice-2026
git submodule update --init --recursive
```

### 3. Enter the prototype

```bash
cd prototype          # you are now inside kavi-prototype
git checkout main     # submodules start on a detached commit; switch to a branch
```

### Keeping in sync

- After `git pull` in the **parent**, the submodule pointer may advance — run
  `git submodule update --recursive` (from the parent) to match it.
- To **update the pointer** after changing the submodule: from the **parent**,
  `git add prototype && git commit` so the new submodule commit is recorded.

### The Android app is not part of this repo

The Android app lives in a **separate private repository** (`kavi-android`) and
is deliberately not vendored here. There is no `android/` submodule in this repo
and `android/` is gitignored, so a clone never populates it. For Android work,
clone `kavi-android` separately (ask the owner for access).

That boundary is enforced rather than just documented: `make check` runs
[`scripts/leak_check.py`](scripts/leak_check.py), which fails when any tracked
file is a private-track type (QNN `.so` libraries, Gradle/Android project files,
app sources, model weights, keystores) or when the tree records a gitlink to the
private repo. Untracked files are out of scope — the check asks what a clone
receives. Run it alone with `make leak-check`.

`adb` ships with the Android **platform-tools**, not with this repo. On Windows,
install platform-tools (or Android Studio) and add `platform-tools\` to `PATH`.
The pinned NDK is r26b (`26.1.10909125`).

## Setup

Run everything from the repository root. The only prerequisite is `uv`
(<https://docs.astral.sh/uv/>).

```bash
make setup        # deps + model weights + FLEURS data + eval manifest
```

`make setup` is the **standard path**: every contributor ends up with the same
pinned artifacts. It needs network access and ~1.5 GB of downloads, and is
equivalent to:

```bash
uv sync           # create .venv from uv.lock (Python 3.12, CPU-only torch)
make models       # pinned Opus-MT vi->en weights + tokenizers  -> models/
make data         # pinned FLEURS vi_vn/en_us test parquets     -> eval_data/raw/
make bench-data   # build the eval manifest                     -> eval_data/eval_manifest_v1.json
```

Offline or bandwidth-constrained? `make setup-min` installs the dependencies and
builds the manifest from the built-in fallback eval set (MT + TTS only, no ASR
items — enough for `make bench` and `make test`).

Everyday commands:

```bash
make check        # ruff lint + format check (read-only; CI-safe)
make fmt          # auto-fix formatting
make test         # pytest (tests/ only)
make bench        # offline harness run (writes bench-results/)
```

## Models & data

Third-party artifacts are **not** committed (see [`NOTICE`](NOTICE)). Their
identity, revision and SHA-256 digests live in
[`assets.lock.toml`](assets.lock.toml), and the fetchers verify against it.

| Command | What it does | Download |
| ------- | ------------ | -------- |
| `make models` | `scripts/fetch_models.py` — `Helsinki-NLP/opus-mt-vi-en` at the pinned revision, converted to CTranslate2 int8 | ~289 MB |
| `make data` | `scripts/fetch_data.py` — `google/fleurs` `vi_vn` + `en_us` test parquets, resumable | ~1.1 GB |
| `make bench-data` | `bench/data_prep.py` — builds the manifest from whatever is in `eval_data/raw/` (offline) | — |
| `make verify-assets` | `scripts/verify_assets.py` — checks every downloaded file against `assets.lock.toml` | — |

Worth knowing:

- `make models` skips `tf_model.h5` (a redundant TensorFlow copy that
  CTranslate2 never reads) and verifies both the upstream checkpoint and the
  converted model.
- Both fetchers are **pinned**. To bump one: change the `revision` in
  `assets.lock.toml`, re-run the fetch, then update the digests reported by
  `make verify-assets`. Never point them at a moving branch.
- `make data` resumes interrupted downloads (`curl -C -`), so re-running is safe.
- `make bench` needs the model weights. Without them the harness does **not**
  abort — `bench/run.py` records an error per item instead. If results look
  empty, run `make verify-assets`.
- `HF_HOME` is honoured by `make models`, so you can relocate the Hugging Face
  cache if `~/.cache/huggingface` is on a small volume.
- The eval set is deterministic (FLEURS sampling is seeded with `42`), so
  results are comparable across machines **provided everyone runs `make setup`
  rather than mixing pinned and unpinned downloads**.

## Workflow

1. Branch from `main`: `git checkout -b feat/<topic>` (or `fix/`, `chore/`).
2. Make focused commits using **Conventional Commits**:
   - `feat:` new feature · `fix:` bugfix · `chore:` maintenance ·
     `docs:` documentation · `refactor:` restructure · `test:` tests.
   - Never write `[skip ci]` (or `[ci skip]`) unless you mean to skip CI: GitHub
     then suppresses **all** workflow runs for that push, changelog included.
3. Keep changes offline/on-device friendly.
4. Open a PR into `main`. Use the PR template.
5. Run `make check` and `make test` before pushing — CI runs both.

## Versioning

Releases follow **Semantic Versioning** (MAJOR.MINOR.PATCH) once the API
stabilizes.

## Prerequisites (detail)

- **Python 3.12** — pinned via `.python-version`; `uv` selects it automatically.
  If it is not installed, run `uv python install 3.12` first.
- **Network for the first `uv sync`** — packages come from PyPI, and
  `torch`/`torchaudio` fetch **CPU-only** wheels from the PyTorch CPU index
  (`https://download.pytorch.org/whl/cpu`, set in `pyproject.toml` under
  `[tool.uv] extra-index-url`). Allow outbound HTTPS to both hosts.
- **Disk** — budget ~2 GB for `.venv`, `models/`, and the FLEURS parquets.

The v0 harness runs **CPU-only** today (faster-whisper / CTranslate2 Opus-MT /
Piper-CPU). On-device QNN candidates land in Phase 4 — see
`docs/reference/phase-4-qnn-plan.md`.

## Tooling & standards

- **Commit messages**: enforced by [`commitlint`](https://commitlint.js.org/)
  via the `commit-msg` pre-commit hook. Titles must follow
  [Conventional Commits](https://www.conventionalcommits.org/). The same rule is
  checked in CI.
- **pre-commit**: install once with `uv run pre-commit install`. It also runs
  basic file hygiene (trailing whitespace, EOF newline, YAML, large files).
- **Linting / formatting**: [`ruff`](https://docs.astral.sh/ruff/) — run
  `make check` (lint + format check) before pushing; `make fmt` to auto-fix.
- **Public boundary**: [`scripts/leak_check.py`](scripts/leak_check.py) — run by
  `make check` (and standalone as `make leak-check`). It asserts that no
  private-track artifact type is tracked. The rules and their rationale live in
  that file; `tests/test_leak_check.py` uses the private repo's real inventory as
  its fixtures.
- **Tests**: `pytest`, configured to collect `tests/` only (`testpaths` in
  `pyproject.toml`) so local vendored trees cannot break the run.
- **Line endings**: governed by [`.gitattributes`](.gitattributes) — LF for
  scripts, `Makefile` and source, regardless of `core.autocrlf`.
- **Changelog**: [`git-cliff`](https://git-cliff.org/) generates `CHANGELOG.md`
  from Conventional Commits. The `Changelog` workflow regenerates it after every
  push to `main` and commits the result. That push needs a **GitHub App** on the
  `main` ruleset bypass list — `github-actions[bot]` cannot be a bypass actor —
  so the workflow mints a short-lived App token rather than using `GITHUB_TOKEN`.
  One-time setup (the `CHANGELOG_SYNC_APP_ID` variable and the
  `CHANGELOG_SYNC_APP_PRIVATE_KEY` secret) is documented in the header comment of
  `.github/workflows/changelog.yml`.
- **CI**: `.github/workflows/ci.yml` runs `make check` and `make test` on
  PRs/pushes to `main`.
- **License**: MIT (see `LICENSE`). Third-party attributions: `NOTICE`.

## Development environments (OS notes)

The core toolchain is **OS-agnostic**: Python + `uv`, `ruff`, and Gradle all run
identically on Windows, Linux, and macOS. The OS-specific pieces are the
**QAIRT conversion toolchain** (Phase 4) and, in the Android repo, the
**Android SDK/NDK install**.

**Per-machine paths have one owner.** `scripts/env.py` resolves the SDK root, the
NDK root, the converter bin dir, the host library dir, the SDK envsetup flavour
and the Python 3.10 converter venv for the current platform. The shell entry
points (`scripts/qairt-env.sh`, `scripts/qairt-env.ps1`) and the conversion
pipeline (`bench/qnn/convert_to_qnn.sh`) consume that resolver instead of
deriving paths themselves, so the four genuinely platform-dependent facts live
in exactly one place:

| fact | POSIX (WSL/Linux) | Windows (Git Bash/PowerShell) |
| --- | --- | --- |
| venv layout | `.venv-qairt/bin` | `.venv-qairt/Scripts` |
| converter bin dir | `bin/x86_64-linux-clang` | `bin/x86_64-windows-msvc` |
| host library dir | `lib/x86_64-linux-clang` (`LD_LIBRARY_PATH`) | `lib/x86_64-windows-msvc` (`PATH`) |
| SDK envsetup | `bin/envsetup.sh` | `bin/envsetup.ps1` |

`uv run python -m scripts.env --check` reports what it resolved and where each
answer came from (argument, environment variable or probe); `--format
bash|powershell|json` emits the environment for a shell or a tool.

**You do not need a hybrid Windows + WSL2 machine — a single OS is enough:**

- **Windows-only** — install Python + `uv`, and (for Phase 4) the Windows QAIRT
  SDK; its converters run natively as `.exe`. No WSL required for the Python
  harness.
- **Linux-only** — same, with the Linux QAIRT SDK (`.qik` installed via
  `qpm-cli`); conversion runs natively.
- **macOS-only** — Python + `uv` work fine. The QAIRT *conversion* toolchain is
  Linux/Windows-only, so run conversion in a VM/container or CI and build the
  app from the committed context binaries.

### Windows specifics

- **`make` is not installed by default.** Install it (`winget install
  ezwinports.make`) or use the Git-for-Windows shell. If you would rather skip
  `make`, every target is a plain `uv run …` command — copy it from the
  `Makefile`.
- **QAIRT conversion on Windows**: dot-source `scripts\qairt-env.ps1`. It creates
  the Python 3.10 `.venv-qairt`, imports the environment resolved by
  `scripts/env.py`, and verifies the converters. The Windows SDK ships the same
  tools with mixed suffixes (`.exe`, `.bat`, and extension-less Python scripts),
  which the resolver's lookup tolerates. `scripts/qairt-env.sh` is the
  Linux/WSL counterpart and deliberately *refuses* to run under Git Bash, where
  `uv` would build a `Scripts/` venv around those same Windows binaries.
- **Environment variables**: `.kavi.env` (bash `export` syntax, for Git
  Bash/WSL) is optional and declares only the two per-machine *roots*. Anything
  derived from them is computed by `scripts/env.py`, and a variable that is set
  but points nowhere — a stale value inherited from Windows — is ignored with a
  warning instead of being used. In PowerShell, dot-source
  `scripts\qairt-env.ps1` or set the variables directly.

**Our setup (reference, not a requirement):** everything runs in **WSL2** on
this box — the Android app is built there against a Linux SDK/NDK (app repo:
`scripts/setup-android-sdk.sh`) and QAIRT conversion uses the **Linux** QAIRT
SDK, whose `x86_64-linux-clang` converters and host libs run natively. The
earlier split (app on Windows, conversion in WSL2) applied while only the
Windows QAIRT SDK was installed; the Linux SDK is now the one in use, so no OS
hopping is required.

**QAIRT conversion toolchain (Phase 4):** host build-time only, needed solely
when producing HTP v73 context binaries. See `docs/reference/phase-4-qnn-plan.md`
§1 for the exact env contract (`QAIRT_SDK_ROOT`, `ANDROID_NDK_ROOT`,
`scripts/env.py`, `qairt-env.sh`/`qairt-env.ps1`, `.venv-qairt`).

## Note on `archive/`

The previous implementation lives under `archive/`. It is reference material,
not active code — do not edit files there; re-implement in the new structure.
