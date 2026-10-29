"""Fetch the pinned third-party Opus-MT artifacts into ``models/``.

One directory pair per ``[models.*]`` entry in ``assets.lock.toml``:

* ``<key>-src/`` - the upstream HuggingFace repository, including the
  SentencePiece tokenizers read at runtime by ``bench/candidates/opusmt_mt.py``.
  ``tf_model.h5`` is skipped: it is a redundant TensorFlow copy that CTranslate2
  never reads (~289 MB).
* ``<key>-ct2/`` - the CTranslate2 int8 conversion used by the harness and the
  on-device build.

Directory names come from the lock entry (``src_dir`` / ``ct2_dir``) so adding a
direction is a data change, not a code change.

Repository, revision and digests come from ``assets.lock.toml``; the resulting
files are verified against it (``--no-verify`` to skip). Neither directory is
tracked by git - see ``NOTICE``.

Needs network access. Run via ``make models``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .assets import load_lock, verify

# Redundant TensorFlow copy; never read by CTranslate2.
_SKIP_PATTERNS = ["tf_model.h5"]


def dirnames(key: str, entry: dict) -> tuple[str, str]:
    """Resolve ``(src_dirname, ct2_dirname)`` for a lock entry."""
    src, ct2 = entry.get("src_dir"), entry.get("ct2_dir")
    if src and ct2:
        return src, ct2
    slug = key.replace("_", "-")
    return f"{slug}-src", f"{slug}-ct2"


def fetch_source(out_dir: Path, repo: str, revision: str, dirname: str) -> Path:
    """Snapshot the upstream HF repo (weights + tokenizers) into ``out_dir``."""
    from huggingface_hub import snapshot_download

    src_dir = out_dir / dirname
    print(f"[fetch_models] downloading {repo}@{revision[:12]} -> {src_dir}")
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=src_dir,
        ignore_patterns=_SKIP_PATTERNS,
    )
    return src_dir


def convert_to_ct2(src_dir: Path, out_dir: Path, dirname: str) -> Path:
    """Convert the Marian checkpoint to CTranslate2 int8."""
    from ctranslate2.converters import TransformersConverter

    ct2_dir = out_dir / dirname
    print(f"[fetch_models] converting {src_dir} -> {ct2_dir} (int8)")
    TransformersConverter(str(src_dir), low_cpu_mem_usage=True).convert(
        str(ct2_dir), quantization="int8", force=True
    )
    return ct2_dir


def _digest_fields(entry: dict) -> bool:
    """True when the entry carries the digests needed to verify."""
    return all(entry.get(k) for k in ("source_sha256", "ct2_sha256"))


def main() -> int:
    models = load_lock()["models"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models"),
        help="Directory to place the fetched artifacts in (default: models).",
    )
    parser.add_argument(
        "--pair",
        action="append",
        metavar="KEY",
        help=(
            "Lock key to fetch, repeatable (default: every [models.*] entry). "
            f"Known: {', '.join(models)}"
        ),
    )
    parser.add_argument(
        "--repo",
        help="Override the upstream repository (single --pair only).",
    )
    parser.add_argument(
        "--revision",
        help="Override the upstream revision (single --pair only).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse the existing source directory; only run the CT2 conversion.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA-256 verification against assets.lock.toml.",
    )
    args = parser.parse_args()

    keys = args.pair or list(models)
    unknown = [k for k in keys if k not in models]
    if unknown:
        parser.error(f"unknown lock key(s): {unknown}; known: {list(models)}")
    if (args.repo or args.revision) and len(keys) != 1:
        parser.error("--repo/--revision only make sense with a single --pair")

    out_dir: Path = args.out
    failed = False

    for key in keys:
        entry = models[key]
        src_name, ct2_name = dirnames(key, entry)
        repo = args.repo or entry["repo"]
        revision = args.revision or entry["revision"]
        print(f"[fetch_models] {key}: {repo}@{revision[:12]}")

        src_dir = out_dir / src_name
        if args.skip_download:
            if not src_dir.exists():
                parser.error(f"--skip-download given but {src_dir} does not exist")
        else:
            src_dir = fetch_source(out_dir, repo, revision, src_name)

        ct2_dir = convert_to_ct2(src_dir, out_dir, ct2_name)

        if args.no_verify or not _digest_fields(entry):
            print(f"[fetch_models] {key}: digests not checked")
            print(f"[fetch_models] {key}: done\n  {src_dir}\n  {ct2_dir}")
            continue

        checks = [
            (
                out_dir / entry["source_path"],
                entry["source_sha256"],
                entry.get("source_size"),
            ),
            (out_dir / entry["ct2_path"], entry["ct2_sha256"], entry.get("ct2_size")),
        ]
        for path, sha, size in checks:
            good, msg = verify(path, sha256=sha, size=size)
            print(f"[fetch_models] {key}: {'PASS' if good else 'FAIL'} {msg}")
            failed = failed or not good

    if failed:
        print(
            "[fetch_models] verification failed - on-disk artifacts do not "
            "match assets.lock.toml",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
