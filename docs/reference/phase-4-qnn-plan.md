# Phase 4 — On-device QNN Conversion & Comparison Plan

> **Status:** Planning (precedes and parallels the Phase-4 implementation, job (1))
> **Superseded scope (2026-08-05):** the ASR (Whisper) and TTS (Piper) QNN paths are **obsolete** — ADR-008 keeps ASR CPU-only (Dual Zipformer via sherpa-onnx); ADR-009 replaces Piper with Supertonic Phase 1 (ADR-024 already deferred Piper QNN). **Live scope = Opus-MT encoder only** (§2, §8, §9). Whisper/Piper sections are retained for reference only.
> **Companion docs:** `benchmarking-todo.md` §Phase 4, `ADR-003` (Hexagon runtime, *Proposed*).
> **Tracking:** issue #51, issue #57 (verified-command corrections), issue #59 (converter output / `.dlc` corrections).

## 0. Purpose

This is the **executable spec** for Phase 4 (on-device QNN). It turns ADR-003's
provisional determination method into concrete, pinned steps:

- how each v0 candidate is converted to a QAIRT / QNN artifact,
- how those artifacts are bundled into the Android app,
- how the on-device comparison runner works, and
- the pass/fail gates that decide whether **QNN becomes the NPU path**.

It is written **before (and in parallel with)** the actual conversion work (job
(1)) so the implementation has a fixed target. The CPU-only baseline already runs
(Phase 2–3), so QNN here is an *optimization pass* layered on top, not a
greenfield build.

---

## 1. Environment contract (host, build-time only)

All conversion happens **off-device** on Linux x86_64 (including WSL) or
Windows with the matching host SDK and toolchain below. Android artifacts
(model `.so` library, HTP v73 context binary, `libQnn*.so` runtime) are committed
so developers can build the app without the SDK. Intermediate model libraries
are host-specific (`.so` on Linux, `.dll` on Windows); byte-identical output
across host operating systems is not guaranteed.

Both hosts are documented as supported, but **neither has been validated
end-to-end yet** — see the caveat at the head of §8.

| Item | Value | Notes |
| --- | --- | --- |
| **QAIRT SDK** | `2.31.0.250130`, Linux or Windows host package at `QAIRT_SDK_ROOT` | **Must match** the device `qnn-2.31` / **HTP v73** runtime. Do **not** upgrade — ABI drift breaks on-device loading. Outside the repo. |
| **ANDROID_NDK_ROOT** | NDK `26.1.10909125` | Matches `archive/`/`sdk.yaml` pin. |
| **Path resolver** | `scripts/env.py` | Single source of truth for the four platform-dependent facts (venv layout, converter bin dir, host library dir, SDK envsetup flavour). Versions and variable names come from `.kavi.yaml`; roots are probed when the variables are unset. `--check` verifies, `--format bash\|powershell\|json` emits, `--converter-bin-dir` answers the pipeline without reading the config. |
| **Linux env helper** | `scripts/qairt-env.sh` (thin adapter over the resolver) | Refuses to run under Git Bash/MSYS, where the Windows converters and a `Scripts/` venv would be the wrong pairing — use WSL2 or `scripts\qairt-env.ps1`. Otherwise creates/activates `.venv-qairt` and `source`s `bin/envsetup.sh` (sets `QNN_SDK_ROOT`, `SNPE_ROOT`). |
| **Converter venv** | Python 3.10 (`.venv-qairt/`, auto-created by the host env helper) | `onnx 1.16.1`, `onnxruntime 1.17.1`, `numpy<2`, `onnx-simplifier`, `scipy`, `lxml`, `absl-py`, `pandas`, `pyyaml`. |
| **Linux converters** | `qnn-onnx-converter`, `qnn-model-lib-generator`, `qnn-context-binary-generator` | Under `$QAIRT_SDK_ROOT/bin/x86_64-linux-clang`. |
| **Windows SDK root / helper** | `$env:QAIRT_SDK_ROOT` (probed: `C:\Qualcomm\…`, then `$HOME\Qualcomm\…`); dot-source `scripts\qairt-env.ps1` | The helper is a thin adapter over `scripts/env.py`: it creates the Python 3.10 venv, imports the resolved environment (including `PYTHONPATH` and `PATH`) and dot-sources the SDK's own `envsetup.ps1`. Run the Visual Studio developer shell first if the SDK needs it. |
| **Windows tools / libraries** | `$env:QAIRT_SDK_ROOT\bin\<windows-host-platform>` and `$env:QAIRT_SDK_ROOT\lib\<windows-host-platform>` | Select the installed platform matching the host architecture (e.g. `x86_64-windows-msvc` or `aarch64-windows-msvc`). Confirm it provides the converter, model-lib generator, `qnn-context-binary-generator.exe` and `QnnHtp.dll`; add its library directory to `PATH` for dependent DLLs. See §8. |
| **Windows compiler** | SDK-compatible Visual Studio C++ build tools and CMake | Build a model DLL for the selected Windows host platform. NDK `26.1.10909125` is for the separate Android model `.so` build. |
| **Device** | Meizu 21 Note — SD 8 Gen 2 (`kalama`), **Android 16 (API 36)**, **HTP v73**, `qnn-2.31` | Runtime **preinstalled**; app **bundles** `libQnn*.so`. |

**Offline guarantee:** the SDK is build-time only; the on-device runtime is
preinstalled firmware + the bundled `.so`. No network at app runtime.

---

## 2. Conversion pipeline (per model)

```text
sourceable model (PyTorch / TFLite / ONNX)
        │
        ▼
  qnn-onnx-converter        →  <model>.cpp        (graph; w8a16 quant: int8 wts / int16 acts)
        │                                   (+ <model>_net.json, QNN_CPU .bin)
        ▼
  qnn-model-lib-generator -c <model>.cpp  →  model library (.so)
        │
        ▼
  qnn-context-binary-generator --htp_arch v73   →  HTP v73 context binary
```

- **Quantization required:** HTP only executes quantized graphs; **w8a16**
  (int8 weights / int16 activations) recommended to protect transformer accuracy.
  HTP silently rejects unquantized graphs.
- **No dynamic shapes:** batch / sequence length must be **fixed** (padding +
  masking). This is the main engineering cost for the autoregressive ASR / MT
  decoders.
- **Artifact choice:** `qnn-onnx-converter` emits `<model>.cpp` (graph source) +
  `<model>_net.json` + a QNN_CPU `.bin` — **not** a `.dlc`. The on-device NPU
  artifact is the **HTP v73 context binary** (plus the model `.so` library), built
  via `qnn-model-lib-generator` + `qnn-context-binary-generator` on a host with
  the QAIRT toolchain — Linux/WSL or Windows. (A
  `.dlc` is a separate SNPE-era format loaded via `libQnnModelDlc.so --dlc_path`
  and is **not** produced by `qnn-onnx-converter`.)

---

## 3. Model-by-model export plan

The current v0 CPU-default candidates are **not QNN-convertible as-is**
(`faster-whisper`/ggml for ASR, `CTranslate2` for MT). QNN converters ingest
**PyTorch / TFLite / ONNX only**, so NPU acceleration requires *re-sourcing* in a
convertible format. Per ADR-003 the stance is **evaluate** (not retire) — the
benchmark below is what decides.

### 3.1 ASR — Whisper Small (244M, MIT) — *evaluate re-source*

> **Superseded (ADR-008):** ASR is CPU-only Dual Zipformer. The Whisper ONNX/QNN path below is retained for reference only.

- **v0 (CPU):** `faster-whisper` Small int8 — ggml, **not** QNN-convertible.
- **For QNN:** export Whisper Small to **ONNX** (e.g. `optimum` Whisper ONNX
  export), **or** use Qualcomm-optimized **Whisper-Small-Quantized** (w8a16) from
  AI Hub Models.
- **Hard part:** the autoregressive decoder needs **fixed-sequence handling** —
  KV-cache / padded decoding, because **no dynamic shapes** are allowed. This is
  the riskiest conversion; budget time for it.
- **Output:** encoder `.cpp` (graph) → host builds model `.so` + HTP v73 context binary.

### 3.2 MT — Opus-MT vi↔en (Helsinki-NLP, PyTorch, Apache-2.0)

- **v0 (CPU):** `CTranslate2`-int8 Opus-MT, **vi→en only** (en→vi weights not in
  repo). CTranslate2 is **not** QNN-convertible.
- **For QNN:** re-export from the **original Helsinki-NLP PyTorch** model → ONNX
  → `qnn-onnx-converter` → `<model>.cpp` (w8a16). No AI Hub prebuilt exists for Opus-MT;
  the en-es recipe is a template.
- **Scope:** export **vi→en** (v0 need) and **en→vi** if the bidirectional eval
  set requires it.
- **Output:** `<model>.cpp` (graph) → HTP v73 context binary (Linux/WSL or Windows).

### 3.3 TTS — Piper (MIT-era `rhasspy/piper`, ONNX) — **Deferred** (ADR-024)

> **ADR-024 supersedes this section.** Piper stays on CPU: cyclic
> graph, `RandomNormalLike` unsupported, already fast on CPU (RTF 0.06–0.22),
> and TTS is not the pipeline bottleneck. The QNN conversion path below is
> retained for reference only.

> **ADR-009 supersedes further (2026-07-30):** v1 TTS is Supertonic Phase 1 (sherpa-onnx `OfflineTts`) → VieNeu-TTS Phase 2, with Piper VITS as fallback. Piper QNN conversion is permanently out of scope.

- **v0 (CPU):** Piper-CPU, **EN leg only** (`en_US-lessac-medium`; `vais1000` VI
  voice not in repo).
- **For QNN (deferred):** Piper is ONNX, but **not a short hop.**
  `en_US-lessac-medium.onnx` contains `RandomNormalLike` (stochastic decoder
  sampling), which `qnn-onnx-converter` 2.31.0.250130 **does not support**.
  Required surgery: replace with deterministic zero-noise (Mul-by-0 of the
  reference tensor), then pin the data-dependent output length by normalizing
  the duration-sum to a fixed `T_FIXED` (see §10).
- **Output (if ever re-evaluated):** `<model>.cpp` (graph) → HTP v73 context
  binary (Linux/WSL or Windows).

---

## 4. Artifact layout & bundling

- **Host-independent outputs** (model `.so` library, v73 context binary, `libQnn*.so` runtime) are
  **committed** so every commit builds — no developer needs the SDK to build.
- **Location in `kavi-android`:**
  - `app/src/main/jniLibs/arm64-v8a/` — `libQnn*.so` (39 `.so` already bundled)
    plus the compiled model `.so` library.
  - `app/src/main/assets/` — model `.so` library + HTP v73 context binaries (read at
    runtime, kept out of `jniLibs` binary load path if preferred).
- **Provenance & commit (ADR-011):** on-device artifacts (HTP v73 context binary, model `.so`, ONNX decoder) are committed **inside `kavi-android` assets**. Host-side `prototype/models/qnn/*` outputs are gitignored (`.gitignore` `models/qnn/*`, commit 3502156) and regenerable — never treat them as deliverables.
- **License:** AI Stack License §1(iv) permits distributing the runtime in object
  code within the app; `public.libraries.txt` does **not** list `libQnn*.so`, so
  the app **bundles** them (ADR-002 / QAIRT gate *ADOPT clean*).

---

## 5. On-device instrumented runner

- **Form:** Android **instrumented test** / thin **service** that reads the
  **same versioned `eval_manifest_v1.json`** the host scorer uses — so QNN and CPU
  candidates are scored on **byte-identical inputs**.
- **Per-utterance logging:** latency, **RTF**, **turnaround (EOS→SA)**,
  **peak RSS**, plus dumped audio/text outputs.
- **Network monitor:** assert **zero network calls** during a run (hard DQ gate).
- **Scoring off-device:** dumps are pulled to the host and scored by
  `bench/scorer.py` (jiwer WER/CER, sacrebleu BLEU; COMET/MOS deferred to Phase
  7) — identical pipeline to the CPU baseline.

---

## 6. Comparison & success gates

For each stage × candidate runtime `{QNN, CPU-default}`, measure RTF,
turnaround, WER/BLEU, peak RAM.

**Hard gates (from `benchmarking-todo.md` §2):**

| Gate | Threshold | Type |
| --- | --- | --- |
| **RTF** | **< 1.0** | Hard |
| **Turnaround** (EOS→SA) | **< 2.0 s** | Hard |
| **No internet** | **0 calls** | Hard (DQ) |

**Per-stage decision rule (ADR-003):** adopt QNN for a stage **iff** it meets the
hard gates **and** beats the CPU baseline on RTF / turnaround **without**
accuracy regression (WER/BLEU within tolerance). Otherwise fall back to CPU for
that stage.

This comparison **closes ADR-003** (→ *Accepted*) and feeds the per-stage picks
that finalize the **tech-stack register**.

---

## 7. Sequencing & risks

- **CPU-first already done** (Phase 2–3): the baseline exists, so a QNN toolchain
  snag cannot block the submission.
- **Conversion order (lowest-risk first):** (1) **Whisper-Small encoder** —
  static `[1,80,3000]`, no decoder loop, and Qualcomm already ships a
  Whisper-Small-Quantized-QNN re-source proving the path; (2) **Opus-MT vi→en** —
  same autoregressive decoder pattern, no sampling op. **(3) Piper — Deferred**
  per ADR-024 (cyclic graph, unsupported ops, already fast on CPU).
  The host runs `qnn-onnx-converter` → `<model>.cpp`, then builds the model
  `.so` + HTP v73 context binary with `qnn-model-lib-generator` /
  `qnn-context-binary-generator` (clang / NDK on Linux/WSL, MSVC / NDK on
  Windows) — verify the installed host tools and libraries against §1 before
  starting.
- **Risks:**
  - ASR decoder **fixed-shape reformulation** (KV-cache / padded decode) — the
    heaviest lift.
  - **QAIRT version must match** the device (`2.31.0.250130` = `qnn-2.31` / HTP
    v73); do not upgrade the SDK.
  - Toolchain env verified (`libc++1` installed; `.venv-qairt` venv runs
    `qnn-onnx-converter`/`qnn-context-binary-generator`).
- **Deferred:** COMET + human MOS depth (Phase 7); pre-ASR denoising gate (Phase
  6).

---

## 8. Verified conversion command reference (QAIRT 2.31.0.250130)

> The Linux flag forms below were previously checked against the installed SDK
> (`qnn-onnx-converter --help`, `qnn-model-lib-generator` source). Checking flag
> forms is **not** the same as completing a conversion: neither host has been
> run end-to-end through steps 2–3 yet. The Windows guidance describes platform
> selection; it has not been executed on Windows either. Check the installed
> 2.31 SDK help for platform-specific options, and treat the **first-validation
> target** below (Whisper encoder, static `[1,80,3000]`) as the first end-to-end
> proof of the Linux-host path.

**Quantization (w8a16, `tf`):** `--param_quantizer tf --act_quantizer tf
--weights_bitwidth 8 --act_bitwidth 16` (all four confirmed present).
`--float_fallback` is **mutually exclusive** with `--input_list` in 2.31.0.250130
(the SDK rejects the combination). Omit it and supply a real `--input_list` (e.g. the
FLEURS mel) for calibration.

**Converter → `.cpp` (+ `.net.json` + QNN_CPU `.bin`) (host: WSL/Linux):**

```bash
qnn-onnx-converter \
  --input_network <model>.onnx \
  --output_path <model>.cpp \
  --input_dim <input_name> "<dims>" \   # repeated per input; singular --input_dim
  --param_quantizer tf --act_quantizer tf \
  --weights_bitwidth 8 --act_bitwidth 16 \
  --input_list <model>_input_list.txt \  # format: "data_file input_name" (data path FIRST)
```

- The converter emits `<model>.cpp` (graph source), `<model>_net.json` (network
  descriptor), and a **QNN_CPU-oriented** `.bin` (context binary — 89 MB for the
  Whisper encoder; calibration artifact, **not** the HTP v73 target). It does
  **not** emit a `.dlc`; the `.dlc` is a separate SNPE-era format (loaded via
  `libQnnModelDlc.so --dlc_path`) and is **not** produced by `qnn-onnx-converter`.
- **No `--output_dim` flag exists** in 2.31 — output dims are inferred from the
  ONNX graph + `input_list`. For data-dependent outputs (Piper TTS), pin the
  length **inside the ONNX graph** (§10), not via a converter flag.

**Model lib + HTP v73 context binary (Linux/WSL reference; NDK `26.1.10909125` for the Android model library):**

```bash
qnn-model-lib-generator -c <model>.cpp -t aarch64-android -n <model> -o <model>_libs/
qnn-model-lib-generator -c <model>.cpp -t x86_64-linux-clang -n <model> -o <model>_libs/
qnn-context-binary-generator \
  --model <model>_libs/x86_64-linux-clang/lib<model>.so \
  --backend "$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
  --htp_arch v73 --binary_file <model>_v73.bin --output_dir <model>_ctx/
```

- `qnn-model-lib-generator` takes `-c <cpp> -t <targets>` only. **`-n <name>` is
  required** — without it the `.so` is named `libqnn_model.so` (SDK default), and
  every downstream `--model` path breaks. The `aarch64-android/` subdir is
  auto-appended inside `-o`.
- `--backend` needs the **full path** to the backend for the platform running
  the generator. Android uses `lib/aarch64-android/libQnnHtp.so`; a Linux x86_64
  generator needs `lib/x86_64-linux-clang/libQnnHtp.so` and an x86_64 model
  library. Windows uses the DLL paths described below.
- `--htp_arch v73` is accepted **only** with `--model <compiled .so>` (passing a
  `.dlc` directly errors with "Unused Arguments").

**Windows context-binary generation:**

1. Select `<windows-host-platform>` from the installed Windows SDK as described
   in §1. Use `bin\<windows-host-platform>\qnn-context-binary-generator.exe`
   with `lib\<windows-host-platform>\QnnHtp.dll` from the same SDK version and
   architecture. If either is absent, use the Linux/WSL toolchain; do not
   substitute an Android or Linux backend.
2. Run the Windows `qnn-onnx-converter` and `qnn-model-lib-generator` from that
   platform's `bin` directory (invoke SDK Python scripts with the Python 3.10
   venv). Build the model for that Windows platform using MSVC/CMake, passing
   the converter's weights file if required. Use the generated **model DLL** as
   `--model`, not the `aarch64-android` model `.so`.
3. Pass the full PowerShell backend path
   `"$env:QAIRT_SDK_ROOT\lib\<windows-host-platform>\QnnHtp.dll"` to
   `--backend`, and put that same library directory on `PATH` so dependent DLLs
   resolve. Configure the HTP v73 target using the installed SDK's supported
   options and write the context binary to the model's context output directory.
   The Android model `.so` and `libQnn*.so` remain separate device artifacts.

Platform layout and Windows build prerequisites are described in Qualcomm's
[Windows setup](https://docs.qualcomm.com/bundle/publicresource/80-63442-50/topics/windows_setup.md)
and [QNN tools reference](https://docs.qualcomm.com/bundle/publicresource/80-63442-50/topics/tools.md);
the installed **2.31.0.250130** SDK remains the authority for available platforms
and flags.

**First validation target = Whisper encoder** (static `[1,80,3000]`, no decoder
loop — lowest risk). Convert, build the model `.so` and generate the context
binary on the same host (WSL/Linux here; Windows uses the DLL workflow above), then run
on-device via
`qnn-net-run --model <model>_libs/aarch64-android/lib<model>.so --backend libQnnHtp.so
--binary_file <model>_v73.bin --input_list input_features:<real_fleurs_mel>.raw`. Save the FP32 `whisper.audio.log_mel_spectrogram`
reference on WSL (`np.save`) so the on-device HTP output can be diffed (max abs
diff should be small, ~int16 step). Confirm profiling shows `BackendType=HTP` for
attention/conv layers, not `CPU`.

## 9. Calibration data (real FLEURS, primary)

FLEURS parquets are available — **use real data;
do not compromise the benchmark with synthetic tensors.**

- **ASR (Whisper):** generate mel spectrograms with `whisper.audio.log_mel_spectrogram`
  (Slaney filterbank + log clamp) — **not** generic `librosa` defaults, or the
  `tf` quantizer ranges won't match on-device preprocessing.
- **MT (Opus-MT):** real VI/EN token sequences from FLEURS text (not random int32
  ranges). Build VI→EN pairs by aligning FLEURS `vi_vn` and `en_us` rows on the
  shared sentence **`id`**.
- **TTS (Piper):** real espeak-ng phonemizations of representative English
  sentences (factory-domain phrase list), not random phoneme IDs. **Deferred**
  per ADR-024 (Piper stays on CPU).
- **Split discipline:** calibration draws from FLEURS **train**; `eval_manifest_v1.json`
  scoring draws from FLEURS **test** (same distribution, no overlap). Teacher-force
  reference transcripts into decoder calibration (not the model's own greedy output).
- **VIVOS** (`AILAB-VNUHCM/vivos`, CC BY-NC-SA) may be used for **VI acoustic
  diversity in calibration only** — it is eval/redistribution-restricted and must
  **never** appear in the shipped `eval_manifest_v1.json`.
- Synthetic `calibration_gen.py` (uniform/Gaussian tensors) stays in the repo
  retained for offline/CI use only (FLEURS parquets are now available).

> **Harness note:** `eval_manifest_v1.json` is currently the 24-item fallback (12
> Opus-MT MT + 12 Piper TTS); ASR + real VI→EN MT items are deferred until FLEURS
> is fetched. Populating it with real FLEURS **test** items is a `data_prep` code
> change, not covered by this doc.

## 10. Pre-execution checklist / known pitfalls

1. ~~**Piper is NOT a short hop.**~~ **Deferred** per ADR-024 — Piper
   stays on CPU. Items 1–3 below are retained for reference if Piper QNN is ever
   re-evaluated.
2. **(Deferred — reference only, see item 1)** Piper output length is data-dependent (duration predictor → length regulator
   → `sum(durations) × hop_length`). Pin it by **normalizing the duration-sum to a
   fixed `T_FIXED`** (insert a `Div` rescale node preserving phoneme ratios) — do
   **not** rely on `onnx-simplifier` with `input_data` alone (it bakes in one
   traced example's timing = misalignment bug). `T_FIXED` is a design constant
   (e.g. 400 latent frames ≈ 4.6 s @ 22050 Hz / hop 256); trim trailing silence
   **outside** the QNN graph.
3. **(Deferred — reference only, see item 1)** `onnx-graphsurgeon` is now in `pyproject.toml` (dependency group `qairt`)
   and installed automatically by `scripts/qairt-env.sh`. The exact `DURATION_OUTPUT_NAME` /
   expand-node names need a **manual Netron inspection** of the patched ONNX first.
4. **WSL validation must `np.save('ref_encoder_out.npy', out[0])`** — the on-device
   diff step reads this FP32 reference; the converter snippet alone doesn't save it.
5. **`qnn-model-lib-generator` flags:** `-n <name>` required (default `.so` =
   `libqnn_model.so`); full `--backend` path; `--htp_arch v73` only with a compiled
   `.so` (see §8).
6. **SDK version lock:** keep `2.31.0.250130` (= device `qnn-2.31` / HTP v73). Do
   not upgrade.
7. **VIVOS CC BY-NC-SA** → calibration only, never in the shipped eval manifest (§9).

## 11. Status / tracking

- **Precedes** job (1) (the actual conversion). Implementation tracks this spec.
  → **2026-08-05:** superseded scope — see banner. Only the Opus-MT encoder path (§2, §8, §9) is live; Whisper/Piper are reference-only.
- **Related:** `benchmarking-todo.md` §Phase 4 (terse checklist), `ADR-003`
  (determination method → this plan), the [open-parameters register](../decisions/README.md#open-parameters).
- **Issues:** #51 (this doc), #52 (ADR-003 tightening), #57 (verified-command
  corrections + pitfalls folded in), #78 (ADR status consolidation — closed).
