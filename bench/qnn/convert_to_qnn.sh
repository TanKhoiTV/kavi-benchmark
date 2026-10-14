#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# QAIRT / QNN conversion wrapper — ONNX → HTP v73 context binary
#
# Converts a model from ONNX to a QNN HTP v73 context binary via the 3-step
# QAIRT pipeline:
#   1. qnn-onnx-converter        →  <model>.cpp (graph + net.json + CPU .bin)
#   2. qnn-model-lib-generator   →  model library (.so for aarch64-android)
#   3. qnn-context-binary-generator  →  HTP v73 context binary
#
# Prerequisites:
#   - QAIRT SDK 2.31.0.250130 installed at $QAIRT_SDK_ROOT
#   - ANDROID_NDK_ROOT set to NDK r26c (26.1.10909125) for model-lib build
#   - Python 3.10 venv (qairt-converters) with onnx 1.16.1, onnxruntime 1.17.1,
#     numpy<2, onnx-simplifier
#   - Ubuntu 22.04 or WSL2 (no GUI libraries needed)
#
# Usage:
#   ./bench/qnn/convert_to_qnn.sh \
#       --onnx models/qnn/opus-mt-vi-en/encoder_model.onnx \
#       --input-name input_ids \
#       --input-dims "1,128" \
#       --input-list /path/to/input_list.txt \
#       --output-dir models/qnn/opus-mt-vi-en/ctx \
#       --name opus_mt_vi_en_encoder \
#       --quantization w8a16
#
# Quantization:
#   --quantization w8a16|w8a8 selects the arm of an A/B bit-width comparison
#   (issue #116). Equivalent to --weights-bitwidth/--act-bitwidth, but matches
#   the `quantization:` key in .kavi.yaml. Builds with different bit-widths
#   MUST use different --name values (suffix _w8a16 / _w8a8); the script refuses
#   to overwrite an existing build that used different settings.
#
# Environment:
#   QAIRT_SDK_ROOT   – path to QAIRT SDK 2.31.0.250130 (required)
#   ANDROID_NDK_ROOT – path to Android NDK r26c (required for step 2)
#   PYTHON           – interpreter used to resolve the converter bin dir
#                      (default python3; the platform mapping lives in
#                      scripts/env.py, so a bare interpreter is enough)
#
# SDK version requirement:
#   Must be 2.31.0.250130 (= device qnn-2.31 / HTP v73).
#   Do NOT upgrade — ABI drift breaks on-device loading.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────────────────────

OUTPUT_DIR="."
NAME="model"
INPUT_LIST=""
INPUT_DIMS_OPT=()
QUANTIZE=true
SKIP_LIB=false
SKIP_CTX=false
VERBOSE=false
FORCE_REBUILD=false

# Set by steps 2 and 3. Declared here because the summary block reads them even
# when those steps are skipped, and `set -u` turns an unset reference into a
# hard failure *after* every artifact has already been written.
MODEL_SO=""
CTX_BIN=""
HTP_BACKEND=""
LIB_SO=""
LIB_OUT=""

# Quantization settings — overridable (issue #116 A/B: w8a16 vs w8a8)
WEIGHTS_BITWIDTH=8
ACT_BITWIDTH=16
PARAM_QUANTIZER=tf
ACT_QUANTIZER=tf
REQUIRE_INPUT_LIST=false
BITWIDTH_FLAGS_USED=false
QUANT_PRESET_USED=false

# ── helpers ──────────────────────────────────────────────────────────────────

die() {
	echo "[ERROR] $*" >&2
	exit 1
}
info() { echo "[INFO] $*"; }

# SHA256 of a file, for stamping the build metadata. Echoes "-" when the file is
# missing or no hashing tool is available (never fails the build).
sha256_of() {
	local f="$1"
	if [[ ! -f "$f" ]]; then
		echo "-"
		return 0
	fi
	if command -v sha256sum >/dev/null 2>&1; then
		sha256sum "$f" | awk '{print $1}'
	elif command -v shasum >/dev/null 2>&1; then
		shasum -a 256 "$f" | awk '{print $1}'
	else
		echo "-"
	fi
	# Always succeed: a hasher that fails must reach the caller's validation
	# and produce a diagnosable error, not abort the script under set -e.
	return 0
}

# Read one key from a build-metadata file. Prints nothing when absent.
meta_value() {
	local key="$1" file="$2"
	[[ -f "$file" ]] || return 0
	sed -n "s/^${key}=//p" "$file" | head -n 1
	return 0
}

usage() {
	cat <<'USAGE'
Usage: convert_to_qnn.sh [OPTIONS]

Required:
  --onnx <path>           Path to the ONNX model file
  --input-name <name>     Name of the model's input tensor (repeatable)
  --input-dims <dims>     Input dimensions in ONNX format, e.g. "1,80,3000"
                          (repeatable, one per --input-name)
  --output-dir <dir>      Output directory for generated artifacts

Conversion options:
  --input-list <path>     Path to calibration input list (for quantization)
  --name <name>           Model name used in generated files (default: basename of ONNX)
  --no-quantize           Skip quantization (float conversion only)

Quantization options (issue #116 — w8a16 vs w8a8 A/B):
  --quantization <w8a16>  Preset selecting the arm, e.g. w8a16 or w8a8.
                          Sets both bit-widths at once. Mutually exclusive with
                          the explicit --weights-bitwidth/--act-bitwidth flags.
  --weights-bitwidth <n>  Weight bit-width: 8 (default) or 16
  --act-bitwidth <n>      Activation bit-width: 8 or 16 (default 16)
  --param-quantizer <q>   Weight quantizer: tf (default), percentile, minmax, entropy
  --act-quantizer <q>     Activation quantizer: tf (default), percentile, minmax, entropy
  --require-input-list    Abort if --input-list is missing instead of falling back
                          to synthesized ranges. Use for any A/B that feeds a
                          recorded decision (ADR-026: calibration must be real data).
  --force-rebuild         Allow overwriting an existing build of the same --name
                          that used different quantization settings.

Safety:
  A build is stamped with <name>_qconv_meta.txt. Re-running the same --name with
  different bit-widths, quantizers or calibration data is refused, so two A/B arms
  cannot silently overwrite each other. Use a distinct --name (_w8a16 / _w8a8).

Pipeline step options:
  --skip-lib              Skip model-lib generation (step 2)
  --skip-ctx              Skip context-binary generation (step 3)

Environment:
  QAIRT_SDK_ROOT          Path to QAIRT SDK 2.31.0.250130
  ANDROID_NDK_ROOT        Path to Android NDK r26c

Examples:
  # Basic conversion (no quantization, onnx input list)
  ./bench/qnn/convert_to_qnn.sh --onnx model.onnx \
      --input-name input_ids --input-dims "1,128" \
      --output-dir out/

  # Full pipeline with quantization
  ./bench/qnn/convert_to_qnn.sh --onnx encoder.onnx \
      --input-name input_features --input-dims "1,80,3000" \
      --input-list calibration_input_list.txt \
      --output-dir out/ --name whisper_encoder

  # Issue #116 A/B — two arms of the same encoder, only bit-width differs.
  # Same --onnx, same --input-list, same SDK; distinct --name per arm.
  for ARM in w8a16 w8a8; do
    ./bench/qnn/convert_to_qnn.sh --onnx encoder_model.onnx \
        --input-name input_ids --input-dims "1,128" \
        --input-list models/qnn/opusmt_input_list.txt \
        --quantization "$ARM" --require-input-list \
        --output-dir models/qnn/opus-mt-vi-en/ctx \
        --name "opus_mt_vi_en_encoder_$ARM"
  done
USAGE
	exit 0
}

# ── parse arguments ──────────────────────────────────────────────────────────

ONNX=""
INPUT_NAMES=()
INPUT_DIMS=()

while [[ $# -gt 0 ]]; do
	case "$1" in
	--onnx)
		ONNX="$2"
		shift 2
		;;
	--input-name)
		INPUT_NAMES+=("$2")
		shift 2
		;;
	--input-dims)
		INPUT_DIMS+=("$2")
		shift 2
		;;
	--input-list)
		INPUT_LIST="$2"
		shift 2
		;;
	--output-dir)
		OUTPUT_DIR="$2"
		shift 2
		;;
	--name)
		NAME="$2"
		shift 2
		;;
	--quantization)
		[[ "$BITWIDTH_FLAGS_USED" == true ]] &&
			die "--quantization conflicts with --weights-bitwidth/--act-bitwidth; use one style or the other"
		if [[ ! "$2" =~ ^w([0-9]+)a([0-9]+)$ ]]; then
			die "--quantization expects w<weights>a<activations>, e.g. w8a16 or w8a8 (got: $2)"
		fi
		WEIGHTS_BITWIDTH="${BASH_REMATCH[1]}"
		ACT_BITWIDTH="${BASH_REMATCH[2]}"
		QUANT_PRESET_USED=true
		shift 2
		;;
	--weights-bitwidth)
		[[ "$QUANT_PRESET_USED" == true ]] &&
			die "--weights-bitwidth conflicts with --quantization; use one style or the other"
		WEIGHTS_BITWIDTH="$2"
		BITWIDTH_FLAGS_USED=true
		shift 2
		;;
	--act-bitwidth)
		[[ "$QUANT_PRESET_USED" == true ]] &&
			die "--act-bitwidth conflicts with --quantization; use one style or the other"
		ACT_BITWIDTH="$2"
		BITWIDTH_FLAGS_USED=true
		shift 2
		;;
	--param-quantizer)
		PARAM_QUANTIZER="$2"
		shift 2
		;;
	--act-quantizer)
		ACT_QUANTIZER="$2"
		shift 2
		;;
	--require-input-list)
		REQUIRE_INPUT_LIST=true
		shift
		;;
	--force-rebuild)
		FORCE_REBUILD=true
		shift
		;;
	--no-quantize)
		QUANTIZE=false
		shift
		;;
	--skip-lib)
		SKIP_LIB=true
		shift
		;;
	--skip-ctx)
		SKIP_CTX=true
		shift
		;;
	--verbose)
		VERBOSE=true
		shift
		;;
	--help | -h) usage ;;
	*) die "Unknown option: $1 (use --help)" ;;
	esac
done

# Validate required arguments
[[ -z "$ONNX" ]] && die "--onnx is required"
[[ ! -f "$ONNX" ]] && die "ONNX file not found: $ONNX"
[[ ${#INPUT_NAMES[@]} -eq 0 ]] && die "At least one --input-name is required"
[[ ${#INPUT_DIMS[@]} -eq 0 ]] && die "At least one --input-dims is required"
[[ ${#INPUT_NAMES[@]} -ne ${#INPUT_DIMS[@]} ]] && die "Number of --input-name entries must match --input-dims entries"

# ── Validate quantization settings ──────────────────────────────────────────

validate_bitwidth() {
	local flag="$1" value="$2"
	[[ "$value" =~ ^[0-9]+$ ]] || die "$flag expects an integer (got: $value)"
	# 4-bit weights and 32-bit activations are out of scope for the #116 A/B and
	# are rejected early rather than producing an off-contract artifact.
	case "$value" in
	8 | 16) ;;
	*) die "$flag must be 8 or 16 for the w8a16/w8a8 A/B (got: $value)" ;;
	esac
}

validate_quantizer() {
	local flag="$1" value="$2"
	[[ -n "$value" ]] || die "$flag must not be empty"
	case "$value" in
	tf | percentile | minmax | entropy) ;;
	*) die "Unknown $flag: '$value'. Valid: tf, percentile, minmax, entropy" ;;
	esac
}

QUANT_TAG="w${WEIGHTS_BITWIDTH}a${ACT_BITWIDTH}"

if $QUANTIZE; then
	validate_bitwidth "--weights-bitwidth" "$WEIGHTS_BITWIDTH"
	validate_bitwidth "--act-bitwidth" "$ACT_BITWIDTH"
	validate_quantizer "--param-quantizer" "$PARAM_QUANTIZER"
	validate_quantizer "--act-quantizer" "$ACT_QUANTIZER"

	if [[ "$ACT_BITWIDTH" == "8" && "$ACT_QUANTIZER" == "tf" ]]; then
		info "  NOTE: 8-bit activations with the 'tf' quantizer — if this QAIRT"
		info "        release rejects the combination, switch both A/B arms to the"
		info "        same alternative (e.g. --act-quantizer percentile) so the"
		info "        only difference between arms stays the bit-width."
	fi

	if $REQUIRE_INPUT_LIST; then
		[[ -n "$INPUT_LIST" ]] || die "--require-input-list: --input-list is required (ADR-026 forbids fallback ranges)"
	fi
	# A mistyped path would otherwise surface as a confusing converter error
	# (or, worse, as a build that silently fell back to no calibration).
	if [[ -n "$INPUT_LIST" && ! -f "$INPUT_LIST" ]]; then
		die "--input-list not found: $INPUT_LIST"
	fi
elif [[ "$BITWIDTH_FLAGS_USED" == true || "$QUANT_PRESET_USED" == true ]]; then
	info "  NOTE: --no-quantize is set; bit-width settings are ignored."
fi

# If NAME is not set explicitly, derive from ONNX basename
if [[ "$NAME" == "." ]]; then
	NAME=$(basename "$ONNX" .onnx)
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Validate SDK environment
[[ -z "${QAIRT_SDK_ROOT:-}" ]] && die "QAIRT_SDK_ROOT is not set. Point it to QAIRT SDK 2.31.0.250130"
QAIRT_SDK_ROOT="$(cd "$QAIRT_SDK_ROOT" 2>/dev/null && pwd)" ||
	die "QAIRT_SDK_ROOT is not a directory: ${QAIRT_SDK_ROOT:-<unset>}"

# The converter bin dir is platform-dependent (bin/x86_64-linux-clang on
# Linux/WSL vs bin/x86_64-windows-msvc on Windows); scripts/env.py owns that
# mapping. This shortcut needs no .kavi.yaml, so a bare python3 suffices.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONVERTER_BIN="$("${PYTHON:-python3}" "$REPO_ROOT/scripts/env.py" --converter-bin-dir)" ||
	die "cannot resolve the converter bin dir (see scripts/env.py --converter-bin-dir)"
CONVERTER="$CONVERTER_BIN/qnn-onnx-converter"
MODEL_LIB_GEN="$CONVERTER_BIN/qnn-model-lib-generator"
CTX_BIN_GEN="$CONVERTER_BIN/qnn-context-binary-generator"

for tool in "$CONVERTER" "$MODEL_LIB_GEN" "$CTX_BIN_GEN"; do
	[[ -x "$tool" ]] || die "Tool not found or not executable: $tool (check QAIRT_SDK_ROOT)"
done

if [[ -n "${ANDROID_NDK_ROOT:-}" ]]; then
	ANDROID_NDK_ROOT="$(cd "$ANDROID_NDK_ROOT" && pwd)"
fi

info "=== QAIRT Conversion Pipeline ==="
info "ONNX:     $ONNX"
info "Name:     $NAME"
info "Output:   $OUTPUT_DIR"
if $QUANTIZE; then
	info "Quant:    $QUANT_TAG (param=$PARAM_QUANTIZER act=$ACT_QUANTIZER)"
else
	info "Quant:    disabled (--no-quantize)"
fi
[[ -n "$INPUT_LIST" ]] && info "Calib:    $INPUT_LIST"
info "SDK:      $QAIRT_SDK_ROOT"
[[ -n "${ANDROID_NDK_ROOT:-}" ]] && info "NDK:      $ANDROID_NDK_ROOT"
info ""

# ── Step 1: ONNX → .cpp (graph) ─────────────────────────────────────────────

CPP_OUT="$OUTPUT_DIR/$NAME"

# ── Build-stamp guard ───────────────────────────────────────────────────────
# Refuse to overwrite an existing build of this --name that used different
# quantization settings, so the two arms of an A/B cannot clobber each other.

META_FILE="$OUTPUT_DIR/${NAME}_qconv_meta.txt"
CALIB_SHA="$(sha256_of "$INPUT_LIST")"

# sha256_of yields "-" for two different things: no file, and no hashing tool.
# A supplied list that cannot be hashed would be stamped exactly like an
# uncalibrated build, and the next calibration-less run would match that stamp
# and overwrite the arm — the hole this guard exists to close. Fail closed.
if [[ -n "$INPUT_LIST" && ! "$CALIB_SHA" =~ ^[0-9a-f]{64}$ ]]; then
	die "Cannot hash the calibration list ($INPUT_LIST): neither sha256sum nor shasum produced a SHA-256. Refusing to stamp a build whose calibration cannot be identified."
fi

if [[ -f "$META_FILE" ]] && ! $FORCE_REBUILD; then
	META_DIFF=()
	# A run that omits --input-list leaves CALIB_SHA="-", so this single
	# comparison covers both changing the calibration list and dropping it: a
	# calibrated build must not be replaced by one on synthesized ranges.
	[[ "$(meta_value input_list_sha256 "$META_FILE")" != "$CALIB_SHA" ]] && META_DIFF+=("input_list_sha256 $(meta_value input_list_sha256 "$META_FILE") → $CALIB_SHA")
	# Recorded because a --no-quantize build otherwise carries the default
	# tf/8/16 values and looks like a quantized build to the next run.
	[[ "$(meta_value quantize "$META_FILE")" != "$QUANTIZE" ]] && META_DIFF+=("quantize $(meta_value quantize "$META_FILE") → $QUANTIZE")
	[[ "$(meta_value param_quantizer "$META_FILE")" != "$PARAM_QUANTIZER" ]] && META_DIFF+=("param_quantizer $(meta_value param_quantizer "$META_FILE") → $PARAM_QUANTIZER")
	[[ "$(meta_value act_quantizer "$META_FILE")" != "$ACT_QUANTIZER" ]] && META_DIFF+=("act_quantizer $(meta_value act_quantizer "$META_FILE") → $ACT_QUANTIZER")
	[[ "$(meta_value weights_bitwidth "$META_FILE")" != "$WEIGHTS_BITWIDTH" ]] && META_DIFF+=("weights_bitwidth $(meta_value weights_bitwidth "$META_FILE") → $WEIGHTS_BITWIDTH")
	[[ "$(meta_value act_bitwidth "$META_FILE")" != "$ACT_BITWIDTH" ]] && META_DIFF+=("act_bitwidth $(meta_value act_bitwidth "$META_FILE") → $ACT_BITWIDTH")
	if [[ ${#META_DIFF[@]} -gt 0 ]]; then
		die "Existing build '$NAME' used different settings:
    - $(printf '%s\n' "${META_DIFF[@]}" | tr '\n' '\n    ')
  Use a distinct --name (e.g. <prefix>_$QUANT_TAG) or pass --force-rebuild."
	fi
fi

info "Step 1: qnn-onnx-converter → $CPP_OUT.cpp"

INPUT_DIM_FLAGS=()
for i in "${!INPUT_NAMES[@]}"; do
	INPUT_DIM_FLAGS+=(--input_dim "${INPUT_NAMES[$i]}" "${INPUT_DIMS[$i]}")
done

QUANT_FLAGS=()
if $QUANTIZE; then
	QUANT_FLAGS=(
		--param_quantizer "$PARAM_QUANTIZER"
		--act_quantizer "$ACT_QUANTIZER"
		--weights_bitwidth "$WEIGHTS_BITWIDTH"
		--act_bitwidth "$ACT_BITWIDTH"
	)
	if [[ -n "$INPUT_LIST" ]]; then
		QUANT_FLAGS+=(--input_list "$INPUT_LIST")
	else
		info "  WARNING: No --input-list provided; quantizer will use fallback ranges."
		info "  For best accuracy, supply real calibration data via --input-list"
		info "  (--require-input-list turns this into a hard error)."
	fi
fi

set -x
"$CONVERTER" \
	--input_network "$ONNX" \
	--output_path "$CPP_OUT" \
	"${INPUT_DIM_FLAGS[@]}" \
	"${QUANT_FLAGS[@]}"
{ set +x; } 2>/dev/null

# Verify outputs
CPP_FILE="$CPP_OUT.cpp"
NET_JSON="$CPP_OUT"_net.json
[[ -f "$CPP_FILE" ]] || die "Step 1 failed: $CPP_FILE not generated"
[[ -f "$NET_JSON" ]] || die "Step 1 failed: $NET_JSON not generated"
info "  ✓ $CPP_FILE"
info "  ✓ $NET_JSON"
info ""

# Stamp the build so a later run with different settings is detected.
cat >"$META_FILE" <<META
# QAIRT conversion build stamp — written by bench/qnn/convert_to_qnn.sh
# Used to detect an accidental rebuild of the same --name with different
# quantization settings. Do not edit by hand; use --force-rebuild to override.
name=$NAME
onnx=$ONNX
quant=$QUANT_TAG
quantize=$QUANTIZE
param_quantizer=$PARAM_QUANTIZER
act_quantizer=$ACT_QUANTIZER
weights_bitwidth=$WEIGHTS_BITWIDTH
act_bitwidth=$ACT_BITWIDTH
input_list=${INPUT_LIST:--}
input_list_sha256=$CALIB_SHA
htp_arch=v73
META
info "  ✓ $META_FILE"
info ""

# ── Step 2: .cpp → model library (.so) ──────────────────────────────────────

if $SKIP_LIB; then
	info "Step 2: SKIPPED (--skip-lib)"
else
	LIB_OUT="$OUTPUT_DIR/${NAME}_libs"
	info "Step 2: qnn-model-lib-generator → $LIB_OUT/aarch64-android/lib${NAME}.so"

	if [[ -z "${ANDROID_NDK_ROOT:-}" ]]; then
		info "  WARNING: ANDROID_NDK_ROOT not set; attempting build with system toolchain."
		info "  Set ANDROID_NDK_ROOT=path/to/ndk-r26c for correct aarch64-android build."
	fi

	set -x
	"$MODEL_LIB_GEN" \
		-c "$CPP_FILE" \
		-t aarch64-android \
		-n "$NAME" \
		-o "$LIB_OUT"
	{ set +x; } 2>/dev/null

	LIB_SO="$LIB_OUT/aarch64-android/lib${NAME}.so"
	[[ -f "$LIB_SO" ]] || die "Step 2 failed: $LIB_SO not generated"
	info "  ✓ $LIB_SO"
	info ""
fi

# ── Step 3: model lib → HTP v73 context binary ──────────────────────────────

if $SKIP_CTX; then
	info "Step 3: SKIPPED (--skip-ctx)"
else
	CTX_OUT="$OUTPUT_DIR/${NAME}_ctx"
	mkdir -p "$CTX_OUT"

	# Use the generated .so from step 2, or accept an external one
	if [[ -f "$LIB_SO" ]]; then
		MODEL_SO="$LIB_SO"
	else
		die "No model .so available for context binary generation (run step 2 or provide one)"
	fi

	HTP_BACKEND="$QAIRT_SDK_ROOT/lib/aarch64-android/libQnnHtp.so"
	[[ -f "$HTP_BACKEND" ]] || die "HTP backend not found: $HTP_BACKEND"

	CTX_BIN="$CTX_OUT/${NAME}_v73.bin"

	info "Step 3: qnn-context-binary-generator → $CTX_BIN"

	set -x
	"$CTX_BIN_GEN" \
		--model "$MODEL_SO" \
		--backend "$HTP_BACKEND" \
		--htp_arch v73 \
		--binary_file "$CTX_BIN" \
		--output_dir "$CTX_OUT"
	{ set +x; } 2>/dev/null

	[[ -f "$CTX_BIN" ]] || die "Step 3 failed: $CTX_BIN not generated"
	info "  ✓ $CTX_BIN"
	info ""
fi

# ── summary ──────────────────────────────────────────────────────────────────

info "=== Conversion complete ==="
if $QUANTIZE; then
	info "Arm:      $QUANT_TAG (param=$PARAM_QUANTIZER act=$ACT_QUANTIZER)"
else
	info "Arm:      float (--no-quantize)"
fi
info "Artifacts in: $OUTPUT_DIR"
info ""
info "Generated files:"
ls -lh "$OUTPUT_DIR"/"$NAME"* 2>/dev/null || true
if [[ -d "${LIB_OUT:-}" ]]; then
	ls -lh "$LIB_OUT"/aarch64-android/ 2>/dev/null || true
fi
if [[ -d "${CTX_OUT:-}" ]]; then
	ls -lh "$CTX_OUT"/ 2>/dev/null || true
fi
info ""
if [[ -n "$MODEL_SO" && -n "$CTX_BIN" ]]; then
	info "On-device verification:"
	info "  qnn-net-run \\"
	info "    --model $MODEL_SO \\"
	info "    --backend $HTP_BACKEND \\"
	info "    --binary_file $CTX_BIN \\"
	info "    --input_list <input_name>:<input_data>.raw"
else
	info "On-device verification: skipped (step 2 and/or 3 did not run)."
fi
