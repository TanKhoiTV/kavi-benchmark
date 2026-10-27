"""Verify downloaded third-party assets against ``assets.lock.toml``.

Checks whatever is present under ``models/`` and ``eval_data/raw/``. Missing
files are reported but are not a hard failure by default, because the offline
eval fallback does not need FLEURS. Use ``--strict`` to require everything.

Exit status is non-zero when a *present* file fails verification, or when
``--strict`` is set and something is missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .assets import load_lock, verify


def main() -> int:
    lock = load_lock()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--data-dir", type=Path, default=Path("eval_data/raw"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail when an expected artifact has not been fetched yet.",
    )
    args = parser.parse_args()

    fleurs = lock["data"]["fleurs"]

    checks: list[tuple[str, Path, str, int | None]] = []
    for key, model in sorted(lock["models"].items()):
        label = key.replace("opus_mt_", "").replace("_", "→")
        checks.append(
            (
                f"{label} source",
                args.models_dir / model["source_path"],
                model["source_sha256"],
                model["source_size"],
            )
        )
        checks.append(
            (
                f"{label} ct2",
                args.models_dir / model["ct2_path"],
                model["ct2_sha256"],
                model["ct2_size"],
            )
        )
    for lang, entry in sorted(fleurs["parquets"].items()):
        checks.append(
            (
                f"fleurs {lang}",
                args.data_dir / entry["local_name"],
                entry["sha256"],
                entry["size"],
            )
        )

    failures: list[str] = []
    missing: list[str] = []
    for label, path, sha, size in checks:
        if not path.exists():
            missing.append(label)
            print(f"  MISSING  {label:14} {path}")
            continue
        good, msg = verify(path, sha256=sha, size=size)
        print(f"  {'PASS' if good else 'FAIL':7}  {label:14} {msg}")
        if not good:
            failures.append(label)

    verified = len(checks) - len(missing) - len(failures)
    print(
        f"\n{verified}/{len(checks)} verified, {len(failures)} failed, "
        f"{len(missing)} missing"
    )
    if failures or (args.strict and missing):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
