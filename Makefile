# Kavi prototype — common tasks
# Run from the repository root.
#
# Every command goes through `uv run`, so you do NOT need an activated
# virtualenv — just have uv installed (https://docs.astral.sh/uv/).
#
# check  vs  fmt:
#   - check is READ-ONLY: it verifies formatting and linting and exits
#     non-zero on any drift. Safe to run in CI. It also asserts the public
#     boundary (no private-track artifacts are tracked) -- see leak-check.
#   - fmt MUTATES files: it reformats the tree and applies ruff autofixes.
#     Do not run it blindly in CI.
#
# setup / models / data  (asset acquisition; all NEED NETWORK on first run):
#   - setup          one-time bootstrap: uv sync + models + data + bench-data.
#   - setup-min      bootstrap without downloads (offline fallback eval set).
#   - models         fetch pinned Opus-MT weights + tokenizers -> models/.
#   - data           fetch pinned FLEURS test parquets         -> eval_data/raw/.
#   - verify-assets  verify downloaded digests against assets.lock.toml.
#
# vad-run  vs  vad-smoke  (VAD-only; no ASR/MT/TTS; needs eval_data/ + assets/noise/):
#   - vad-run runs ONE (THRESHOLD, TIMEOUT_MS) configuration over every item in
#     eval_data/vad_manifest_v2.json. Override: make vad-run THRESHOLD=0.1 TIMEOUT_MS=350
#   - vad-smoke runs the ADR default configuration on 1 VI + 1 EN item per
#     condition x SNR x pause cell. It is NOT the sweep.
#
# bench  vs  bench-data:
#   - bench runs a lightweight, OFFLINE --smoke pass by default and writes
#     results to bench-results/. This matches the project's offline /
#     on-device-at-runtime requirement.
#   - bench-data builds the eval manifest (eval_data/eval_manifest_v1.json)
#     from whatever is in eval_data/raw/ -- offline, with a built-in fallback
#     (MT + TTS only) when the FLEURS parquets are absent.

.PHONY: install setup setup-min check fmt test bench bench-data beam-sweep beam-sweep-smoke models data verify-assets leak-check vad-run vad-smoke vad-bounds vad-manifest vad-gold-audio vad-collar vad-audit vad-sweep vad-sweep-smoke vad-manifest-40 vad-collar-sensitivity vad-stage5 vad-decision

install: ## Install dependencies
	uv sync

setup: ## One-time bootstrap: deps + model weights + FLEURS data + manifest (NEEDS NETWORK)
	uv sync
	$(MAKE) models
	$(MAKE) data
	$(MAKE) bench-data

setup-min: ## Bootstrap without downloads (offline fallback eval set)
	uv sync
	$(MAKE) bench-data

check: ## Read-only lint/format check (CI-safe; verifies, no mutation)
	uv run ruff check .
	uv run ruff format --check .
	uv run python -m scripts.check_docs
	uv run pre-commit run yamllint --all-files
	$(MAKE) leak-check

leak-check: ## Assert the public boundary: no private-track artifacts are tracked
	uv run python -m scripts.leak_check

fmt: ## Format + autofix (mutates files)
	uv run ruff format .
	uv run ruff check --fix .

test: ## Run the test suite (pytest)
	uv run pytest

bench: ## Offline harness run (--smoke by default; writes bench-results/)
	uv run python -m bench.run --smoke --out bench-results

bench-data: ## Build eval manifest from local data (offline; fallback if absent)
	uv run python -m bench.data_prep --out eval_data/eval_manifest_v1.json

beam-sweep: ## Formal MT beam sweep: repeats + warmup, both BLEU definitions
	uv run python -m bench.beam_sweep \
		--manifest eval_data/mt_vi_en_eval_manifest.json \
		--out bench-results/beam-sweep-mt \
		--beams 1 2 4 5 8 --repeats 3 --warmup 1 \
		--report docs/reference/beam-sweep-mt.md

beam-sweep-smoke: ## Wiring check: 5 items, 1 repeat (NOT the result)
	uv run python -m bench.beam_sweep \
		--manifest eval_data/mt_vi_en_eval_manifest.json \
		--out bench-results/beam-sweep-smoke \
		--beams 1 4 --repeats 1 --warmup 1 --limit 5

models: ## Fetch pinned Opus-MT weights + tokenizers (NEEDS NETWORK, ~290 MB)
	uv run python -m scripts.fetch_models --out models

data: ## Fetch pinned FLEURS test parquets (NEEDS NETWORK, ~1.1 GB, resumable)
	uv run python -m scripts.fetch_data --workdir eval_data

verify-assets: ## Verify downloaded models/data against assets.lock.toml
	uv run python -m scripts.verify_assets

THRESHOLD ?= 0.05
TIMEOUT_MS ?= 500

vad-run: ## One energy-VAD configuration over the full VAD manifest (THRESHOLD, TIMEOUT_MS)
	uv run python -m bench.vad_runner --threshold $(THRESHOLD) --timeout-ms $(TIMEOUT_MS) --out bench-results/vad

vad-smoke: ## VAD smoke test: default config, 1 VI + 1 EN item per cell (not the sweep)
	uv run python -m bench.vad_runner --per-cell-per-lang 1 --out bench-results/vad-smoke

LABELS ?= eval_data/vad_gold_labels.csv

vad-bounds: ## Reference speech bounds (Silero) for every FLEURS clip -> bench/vad_source_bounds.json
	uv run python -m scripts.compute_source_bounds

vad-manifest: ## Regenerate eval_data/vad_manifest_v2.json (needs bench/vad_source_bounds.json)
	uv run python -c "from bench.vad_manifest import build_vad_manifest; build_vad_manifest()"

vad-gold-audio: ## Render the 50 gold clips + blank label sheet for hand labelling
	uv run python -m scripts.render_gold_audio

vad-collar: ## Stage 3B: reference-vs-hand boundary error and collar -> bench/vad_collar.json (LABELS=...)
	uv run python -m bench.vad_label_error --labels $(LABELS)

vad-audit: ## Audit eval_data/vad_manifest_v2.json and the gold template against the Stage 3A spec
	uv run python -m scripts.audit_vad_manifest

MANIFEST ?= eval_data/vad_manifest_v2.json

vad-sweep: ## Full threshold x timeout sweep (Stage 1 grid) over MANIFEST; audio built once
	uv run python -m bench.vad_sweep --manifest $(MANIFEST) --out bench-results/vad-sweep

vad-sweep-smoke: ## Same sweep on 1 VI + 1 EN item per cell (quick check, not the result)
	uv run python -m bench.vad_sweep --manifest $(MANIFEST) --per-cell-per-lang 1 --out bench-results/vad-sweep-smoke

vad-manifest-40: ## Borderline follow-up: 40 items per cell -> eval_data/vad_manifest_v2_n20.json
	uv run python -c "from bench.vad_manifest import build_vad_manifest; build_vad_manifest(n_per_lang=20, out_path='eval_data/vad_manifest_v2_n20.json')"

vad-collar-sensitivity: ## Score the 5x5 sweep at several collars (no hand labels needed)
	uv run python -m bench.vad_collar_sensitivity --manifest $(MANIFEST) --out bench-results/vad-collar-sensitivity

vad-stage5: ## Stage 5: full sweep with the Stage 3B collar -> bench/results/vad/ (table + record), raw in bench-results/
	uv run python -m bench.vad_stage5

vad-decision: ## Stage 6 helper: apply the criteria to every pair and cell (writes bench-results/ only)
	uv run python -m bench.vad_decision
