"""Submit the MT encoder to Qualcomm AI Hub for compilation and HTP profiling.

Dev-time tooling only: requires `uv pip install qai-hub` and a configured API
token (`qai-hub configure --api_token <token>`). Nothing here runs at app
runtime, and no project dependency is added for it.

Why this exists
---------------
On-device QNN execution cannot be driven from an `adb` shell: SELinux denies the
`shell` domain access to `/dev/adsprpc-smd` (vendor_qdsp_device), so every HTP
run needs an app context. AI Hub runs the model inside Qualcomm's own harness,
where that restriction does not apply, and reports per-layer timings on
Snapdragon 8 Gen 2 silicon - the target SoC.

Usage
-----
    uv pip install qai-hub
    qai-hub configure --api_token "$QAI_HUB_API_TOKEN"
    uv run python bench/aihub_profile.py

The compile job targets a QNN context binary; the profile job reports latency
per layer. Results are written to `bench-results/aihub-encoder/`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# AI Hub's upload progress bars swamp the log for a 186 MB model.
os.environ.setdefault("TQDM_DISABLE", "1")

DEFAULT_OUT = "bench-results/aihub-encoder"
SEQ_LEN = 128
SD8G2 = "sm8550"


def resolve_device(hub_mod, needle: str):
    """Pick the first device matching ``needle`` in its name or attributes.

    AI Hub device names do not carry the SoC ("Samsung Galaxy S23"); the SoC is
    an attribute string such as ``chipset:sm8550`` or ``hexagon:v73``. Matching
    on attributes is what makes ``--device sm8550`` mean the target silicon.
    """
    devices = hub_mod.get_devices()
    wanted = needle.lower()
    matches = [
        device
        for device in devices
        if wanted in device.name.lower()
        or any(wanted in str(attr).lower() for attr in device.attributes)
    ]
    if not matches:
        names = "\n  ".join(sorted(device.name for device in devices))
        raise SystemExit(f"no device matching {needle!r}; available:\n  {names}")
    return matches[0]


def load_calibration(directory: Path, limit: int = 20) -> dict[str, list[np.ndarray]]:
    """Read ``ids_<n>.raw`` / ``mask_<n>.raw`` pairs into AI Hub dataset entries."""
    ids_paths = sorted(directory.glob("ids_*.raw"))[:limit]
    if not ids_paths:
        raise SystemExit(f"no calibration raws in {directory}")

    ids_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    for ids_path in ids_paths:
        mask_path = ids_path.with_name(ids_path.name.replace("ids_", "mask_"))
        # AI Hub validates calibration entries against input_specs, so the dtype
        # here must match the spec (int64), and the leading dim is batch-first.
        ids_rows.append(
            np.fromfile(ids_path, dtype=np.int32).reshape(1, SEQ_LEN).astype(np.int64)
        )
        mask_rows.append(
            np.fromfile(mask_path, dtype=np.int32).reshape(1, SEQ_LEN).astype(np.int64)
        )

    return {"input_ids": ids_rows, "attention_mask": mask_rows}


def main() -> int:
    # The Windows console defaults to cp932, which cannot encode the emoji and
    # dashes qai-hub prints in its own progress output.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--onnx",
        required=True,
        help="exported encoder ONNX, e.g. ../kavi-android/models/qnn/opus-mt-vi-en/encoder_model.onnx",
    )
    parser.add_argument(
        "--calib",
        default=None,
        help="dir of ids_<n>.raw / mask_<n>.raw pairs; required unless --fp16",
    )
    parser.add_argument(
        "--device",
        default=SD8G2,
        help="device name or attribute substring; default targets sm8550 (SD8G2)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="result directory")
    parser.add_argument("--name", default="opus-mt-vi-en-encoder", help="model name")
    parser.add_argument(
        "--model-id",
        default=None,
        help="reuse an already-uploaded model instead of re-uploading the ONNX",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="reuse an already-uploaded calibration dataset",
    )
    parser.add_argument(
        "--compile-job-id",
        default=None,
        help="resume from an already-submitted compile job",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="skip the calibration set, compiling fp16 instead of w8a16",
    )
    args = parser.parse_args()

    try:
        import qai_hub as hub
    except ImportError:
        sys.stderr.write("qai-hub is not installed: uv pip install qai-hub\n")
        return 1

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX not found: {onnx_path}")

    device = resolve_device(hub, args.device)
    print(f"device: {device.name}")

    model = (
        hub.get_model(args.model_id)
        if args.model_id
        else hub.upload_model(model=str(onnx_path), name=args.name)
    )
    print(f"model: {model.model_id}")

    # Form is ((shape), "dtype"). The dtype must match what AI Hub infers from
    # the ONNX (int64 here), while the calibration set stays float32 - the two
    # serve different stages. --truncate_64bit_io then casts the IO to int32
    # for HTP, which has no 64-bit integer tensors.
    input_specs = {
        "input_ids": ((1, SEQ_LEN), "int64"),
        "attention_mask": ((1, SEQ_LEN), "int64"),
    }
    calibration_data = None
    if not args.fp16:
        if args.dataset_id:
            calibration_data = hub.get_dataset(args.dataset_id)
            print(f"calibration set: {calibration_data.dataset_id}")
        else:
            if args.calib is None:
                raise SystemExit("--calib is required unless --fp16 is passed")
            calibration = load_calibration(Path(args.calib))
            calibration_data = hub.upload_dataset(
                calibration, name=f"{args.name}-calib"
            )
            print(f"uploaded calibration set ({len(calibration['input_ids'])} samples)")

    # AI Hub accepts onnx / precompiled_qnn_onnx / qnn_dlc / tflite. The
    # QNN-specific runtime is what yields an HTP context binary. Supplying
    # calibration data obliges us to name the quantisation explicitly; w8a16
    # is what ADR-003 specifies (8-bit weights, 16-bit activations).
    # --truncate_64bit_io is required because the exported graph declares
    # int64 input_ids/attention_mask and HTP has no 64-bit integer tensors.
    compile_options = ["--target_runtime precompiled_qnn_onnx", "--truncate_64bit_io"]
    if calibration_data is not None:
        compile_options.append("--quantize_full_type w8a16")

    if args.compile_job_id:
        resume_job = hub.get_job(args.compile_job_id)
        if not isinstance(resume_job, hub.CompileJob):
            raise SystemExit(
                f"{args.compile_job_id} is a {type(resume_job).__name__}, not a compile job"
            )
        compile_job = resume_job
        print(f"compile job (resumed): {compile_job.job_id} - {compile_job.url}")
    else:
        compile_job = hub.submit_compile_job(
            model=model,
            device=device,
            name=f"{args.name}-compile",
            input_specs=input_specs,
            options=" ".join(compile_options),
            calibration_data=calibration_data,
        )
        print(f"compile job: {compile_job.job_id} - {compile_job.url}")
    compile_job.wait()
    if compile_job.get_status().failure:
        print(f"compile failed: {compile_job.get_status().message}")
        return 1

    target_model = compile_job.get_target_model()
    if target_model is None:
        print("compile job produced no target model")
        return 1
    print(f"compiled model: {target_model.model_id}")

    # No options: the default profile already reports per-layer execution time,
    # and '--profile_level detailed' is not a recognised argument here.
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device,
        name=f"{args.name}-profile",
    )
    print(f"profile job: {profile_job.job_id} - {profile_job.url}")
    profile_job.wait()
    if profile_job.get_status().failure:
        print(f"profile failed: {profile_job.get_status().message}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = profile_job.download_profile()
    (out_dir / "profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )

    execution = profile.get("execution_summary", {})
    print(json.dumps(execution, indent=2))
    summary = {
        "device": device.name,
        "quantization": "fp16" if args.fp16 else "w8a16",
        "model_id": target_model.model_id,
        "compile_job": compile_job.job_id,
        "profile_job": profile_job.job_id,
        "execution_summary": execution,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
