"""Fetch the pinned FLEURS test parquets into ``eval_data/raw/``.

Downloads the ``google/fleurs`` test split for every language listed in
``assets.lock.toml`` (``vi_vn`` and ``en_us`` by default), pinned to the
recorded dataset revision, and verifies SHA-256 + size afterwards.

The parquets are large (~691 MB + ~402 MB), so downloads are resumable
(HTTP range / ``curl -C -``); a dropped connection continues from the offset
instead of restarting.

Needs network access. Run via ``make data``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .assets import load_lock, verify


def _curl(url: str, out: Path, max_retries: int) -> bool:
    """Download ``url`` to ``out`` with resume; return True on success."""
    for attempt in range(1, max_retries + 1):
        print(f"[fetch_data] {out.name}: attempt {attempt}/{max_retries}")
        rc = subprocess.run(
            [
                "curl",
                "-sSL",
                "--retry",
                "20",
                "--retry-delay",
                "2",
                "--max-time",
                "600",
                "-C",
                "-",
                "-o",
                str(out),
                url,
            ],
            check=False,
        ).returncode
        if rc == 0 and out.exists() and out.stat().st_size > 1_000_000:
            return True
        print(f"[fetch_data]   incomplete (curl rc={rc}); resuming")
    return False


def main() -> int:
    lock = load_lock()["data"]["fleurs"]
    base_url = lock["base_url"].format(revision=lock["revision"])
    parquets = lock["parquets"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("eval_data"),
        help="Directory that receives raw/<parquet> (default: eval_data).",
    )
    parser.add_argument(
        "--lang",
        action="append",
        choices=sorted(parquets),
        help="Limit to these languages (repeatable; default: all in the lock).",
    )
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA-256 verification (not recommended).",
    )
    args = parser.parse_args()

    langs = args.lang or sorted(parquets)
    raw_dir = args.workdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_data] google/fleurs @ {lock['revision'][:12]} -> {raw_dir}")

    ok = True
    for lang in langs:
        entry = parquets[lang]
        out = raw_dir / entry["local_name"]
        url = f"{base_url}/{entry['path']}"

        needs_download = not out.exists() or out.stat().st_size != entry["size"]
        if needs_download and not _curl(url, out, args.max_retries):
            print(f"[fetch_data] FAIL download: {lang}", file=sys.stderr)
            ok = False
            continue

        if args.no_verify:
            continue
        good, msg = verify(out, sha256=entry["sha256"], size=entry["size"])
        print(f"[fetch_data] {'PASS' if good else 'FAIL'} {msg}")
        ok = ok and good

    if not ok:
        print("[fetch_data] one or more parquets failed", file=sys.stderr)
        return 1
    print(f"[fetch_data] done -> {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
