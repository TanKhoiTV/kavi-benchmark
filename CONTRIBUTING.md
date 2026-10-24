# Contributing to Kavi (kavi-benchmark)

## Repository role

This is the **implementation** submodule of `aivoice-2026` (the umbrella repo). It contains the code and internal docs; public-facing documentation belongs in the parent. Third-party model weights and eval corpora are **not** committed — they are fetched on demand (see [Models & data](#models--data)).

## Cloning

`kavi-benchmark` is a **Git submodule** of the public parent repo [`aivoice-2026`](https://github.com/TanKhoiTV/aivoice-2026). That means it does **not** come down with a normal `git clone` of the parent. Both repositories are public, so no access request is needed.

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
cd prototype          # you are now inside kavi-benchmark
git checkout main     # submodules start on a detached commit; switch to a branch
```

### Keeping in sync

- After `git pull` in the **parent**, the submodule pointer may advance — run `git submodule update --recursive` (from the parent) to match it.
- To **update the pointer** after changing the submodule: from the **parent**, `git add prototype && git commit` so the new submodule commit is recorded.

### The Android app is not part of this repo

The Android app lives in a **separate private repository** (`kavi-android`) and is deliberately not vendored here. There is no `android/` submodule in this repo and `android/` is gitignored, so a clone never populates it. For Android work, clone `kavi-android` separately (ask the owner for access).

That boundary is enforced rather than just documented: `make check` runs [`scripts/leak_check.py`](scripts/leak_check.py), which fails when any tracked file is a private-track type (QNN `.so` libraries, Gradle/Android project files, app sources, model weights, keystores), or when the tree records a gitlink to the private repo. Untracked files are out of scope: the check asks what a clone receives. Run it alone with `make leak-check`.

## Setup

Run everything from the repository root. The only prerequisite is `uv` (<https://docs.astral.sh/uv/>).

```bash
make setup        # deps + model weights + FLEURS data + eval manifest
```

`make setup` is the **standard path**: every contributor ends up with the same pinned artifacts. It needs network access and ~1.5 GB of downloads, and is equivalent to:

```bash
uv sync           # create .venv from uv.lock (Python 3.12, CPU-only torch)
make models       # pinned Opus-MT vi->en weights + tokenizers  -> models/
make data         # pinned FLEURS vi_vn/en_us test parquets     -> eval_data/raw/
make bench-data   # build the eval manifest                     -> eval_data/eval_manifest_v1.json
```

Offline or bandwidth-constrained? `make setup-min` installs the dependencies and builds the manifest from the built-in fallback eval set (MT + TTS only, no ASR items — enough for `make bench` and `make test`).

Everyday commands:

```bash
make check        # ruff lint + format check (read-only; CI-safe)
make fmt          # auto-fix formatting
make test         # pytest (tests/ only)
make bench        # offline harness run (writes bench-results/)
```

## Models & data

Third-party artifacts are **not** committed (see [`NOTICE`](NOTICE)). Their identity, revision and SHA-256 digests live in [`assets.lock.toml`](assets.lock.toml), and the fetchers verify against it.

| Command | What it does | Download |
| ------- | ------------ | -------- |
| `make models` | `scripts/fetch_models.py` — `Helsinki-NLP/opus-mt-vi-en` at the pinned revision, converted to CTranslate2 int8 | ~289 MB |
| `make data` | `scripts/fetch_data.py` — `google/fleurs` `vi_vn` + `en_us` test parquets, resumable | ~1.1 GB |
| `make bench-data` | `bench/data_prep.py` — builds the manifest from whatever is in `eval_data/raw/` (offline) | — |
| `make verify-assets` | `scripts/verify_assets.py` — checks every downloaded file against `assets.lock.toml` | — |

Worth knowing:

- `make models` skips `tf_model.h5` (a redundant TensorFlow copy that CTranslate2 never reads) and verifies both the upstream checkpoint and the converted model.
- Both fetchers are **pinned**. To bump one: change the `revision` in `assets.lock.toml`, re-run the fetch, then update the digests reported by `make verify-assets`. Never point them at a moving branch.
- `make data` resumes interrupted downloads (`curl -C -`), so re-running is safe.
- `make bench` needs the model weights. Without them the harness does **not** abort — `bench/run.py` records an error per item instead. If results look empty, run `make verify-assets`.
- `HF_HOME` is honoured by `make models`, so you can relocate the Hugging Face cache if `~/.cache/huggingface` is on a small volume.
- The eval set is deterministic (FLEURS sampling is seeded with `42`), so results are comparable across machines **provided everyone runs `make setup` rather than mixing pinned and unpinned downloads**.

## Workflow

1. Branch from `main`: `git checkout -b feat/<topic>` (or `fix/`, `chore/`).
2. Make focused commits using **Conventional Commits**:
   - `feat:` new feature · `fix:` bugfix · `chore:` maintenance · `docs:` documentation · `refactor:` restructure · `test:` tests.
   - Never write `[skip ci]` (or `[ci skip]`) unless you mean to skip CI: GitHub then suppresses **all** workflow runs for that push, changelog included.
3. Keep changes offline/on-device friendly.
4. Open a PR into `main`. Use the PR template.
5. Run `make check` and `make test` before pushing — CI runs both.

## Versioning

Releases follow **Semantic Versioning** (MAJOR.MINOR.PATCH) once the API stabilizes.

## Prerequisites (detail)

- **Python 3.12** — pinned via `.python-version`; `uv` selects it automatically. If it is not installed, run `uv python install 3.12` first.
- **Network for the first `uv sync`** — packages come from PyPI, and `torch`/`torchaudio` fetch **CPU-only** wheels from the PyTorch CPU index (`https://download.pytorch.org/whl/cpu`, declared in `pyproject.toml` as an `explicit` index used only by the torch packages). Allow outbound HTTPS to both hosts.
- **Disk** — budget ~2 GB for `.venv`, `models/`, and the FLEURS parquets.

The harness runs **CPU-only** (faster-whisper / CTranslate2 Opus-MT / Piper-CPU). On-device QNN work — candidates, conversion, context binaries — is out of scope here and lives in the private `kavi-android` repo.

## Tooling & standards

- **Commit messages**: enforced by [`commitlint`](https://commitlint.js.org/) via the `commit-msg` pre-commit hook. Titles must follow [Conventional Commits](https://www.conventionalcommits.org/). The same rule is checked in CI.
- **pre-commit**: install once with `uv run pre-commit install`. It also runs basic file hygiene (trailing whitespace, EOF newline, YAML, large files).
- **Linting / formatting**: [`ruff`](https://docs.astral.sh/ruff/) — run `make check` (lint + format check) before pushing; `make fmt` to auto-fix.
- **Public boundary**: [`scripts/leak_check.py`](scripts/leak_check.py) — run by `make check` (and standalone as `make leak-check`). It asserts that no private-track artifact type is tracked. The rules and their rationale live in that file; `tests/test_leak_check.py` uses the private repo's real inventory as its fixtures.
- **Tests**: `pytest`, configured to collect `tests/` only (`testpaths` in `pyproject.toml`) so local vendored trees cannot break the run.
- **Line endings**: governed by [`.gitattributes`](.gitattributes) — LF for scripts, `Makefile` and source, regardless of `core.autocrlf`.
- **Changelog**: [`git-cliff`](https://git-cliff.org/) generates `CHANGELOG.md` from Conventional Commits. The `Changelog` workflow regenerates it after every push to `main` and commits the result. That push needs a **GitHub App** on the `main` ruleset bypass list — `github-actions[bot]` cannot be a bypass actor — so the workflow mints a short-lived App token rather than using `GITHUB_TOKEN`. One-time setup (the `CHANGELOG_SYNC_APP_ID` variable and the `CHANGELOG_SYNC_APP_PRIVATE_KEY` secret) is documented in the header comment of `.github/workflows/changelog.yml`.
- **CI**: `.github/workflows/ci.yml` runs `make check` and `make test` on PRs/pushes to `main`.
- **License**: MIT (see `LICENSE`). Third-party attributions: `NOTICE`.

## Development environments (OS notes)

The harness is **OS-agnostic**: Python + `uv`, `ruff` and `pytest` behave the same on Windows, Linux and macOS, and the benchmark runs fully offline once the pinned assets are fetched. There is no host-specific configuration left in this repo — the QAIRT conversion toolchain, and with it the SDK/NDK path resolution, moved to the private `kavi-android` repo in the 2026-10 re-organisation.

### Windows specifics

**`make` is not installed by default.** Install it (`winget install ezwinports.make`) or use the Git-for-Windows shell. If you would rather skip `make`, every target is a plain `uv run …` command — copy it from the [`Makefile`](Makefile).

### On-device work

Building the app, running the QAIRT converters and producing HTP v73 context binaries is out of scope for this repository. Those live in the private `kavi-android` repo: its `CONTRIBUTING.md` covers the Android SDK/NDK install and `scripts/setup-android-sdk.sh`, and `docs/ndk-conversion-runbook.md` there covers conversion.

## Note on the previous implementation

It lives in `aivoice-2026/archive/` — reference material, not active code. Do not edit files there; re-implement in the current structure.
