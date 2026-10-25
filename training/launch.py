"""Orchestrate the eight T4 lanes across four Kaggle accounts.

Kaggle has no cross-account scheduling API, so this drives the `kaggle` CLI once
per account with that account's own credentials, and keeps a local ledger of what
was pushed where. Nothing here holds a token: credentials live in
``training/accounts.toml`` (gitignored) and are written to a per-account
``KAGGLE_CONFIG_DIR`` at run time, never into a kernel and never into git.

    python -m training.launch check                 # validate accounts + plan
    python -m training.launch dataset               # stage + upload the input dataset (per account)
    python -m training.launch run --wave 1          # one lane per account
    python -m training.launch run --wave 2          # the second lane per account
    python -m training.launch status
    python -m training.launch pull --lane mix-lenp05
    python -m training.launch probe --account m1    # measure the quota counter

`--dry-run` works before or after the subcommand: it stages the files and prints
the CLI calls without executing them, which is also how the notebook layout is
reviewed without touching Kaggle.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .lanes import ACCOUNTS, LANES, schedule

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "eval_data" / "kaggle"
LEDGER = WORK / "ledger.json"
DATASET_TITLE = "kavi mt finetune vi-en"
DATASET_SLUG = DATASET_TITLE.replace(" ", "-")

# Lane config key -> training CLI flag. Anything not listed is not passed, so a
# lane cannot silently set a flag the trainer does not have.
LANE_ARG = {
    "lr": "--lr",
    "epochs": "--epochs",
    "length_penalty": "--length-penalty",
    "batch_size": "--batch-size",
    "max_steps": "--max-steps",
}


def read_accounts() -> dict[str, dict]:
    """Parse training/accounts.toml without a TOML dependency."""
    path = REPO / "training" / "accounts.toml"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\ncopy training/accounts.toml.example and fill in the four\n"
            "Kaggle username/key pairs (Kaggle -> Account -> Create New Token)."
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    accounts: dict[str, dict] = {}
    current: dict = {}
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            if current:
                accounts[current["name"]] = current
            current = {}
            continue
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        current[key] = value.strip().strip('"').strip("'")
    if current:
        accounts[current["name"]] = current
    missing = [name for name in ACCOUNTS if name not in accounts]
    if missing:
        raise SystemExit(f"accounts.toml is missing: {', '.join(missing)}")
    return accounts


def config_dir(account: dict) -> Path:
    """Write that account's kaggle.json to an isolated config dir (gitignored)."""
    directory = WORK / "creds" / account["name"]
    directory.mkdir(parents=True, exist_ok=True)
    cfg = directory / "kaggle.json"
    cfg.write_text(json.dumps({"username": account["username"], "key": account["key"]}))
    # Windows has ACLs, not mode bits; a failed chmod there is not an error.
    with contextlib.suppress(OSError):
        cfg.chmod(0o600)
    return directory


def kaggle(account: dict, *args: str, dry: bool = False, capture: bool = False) -> str:
    env = dict(os.environ)
    env["KAGGLE_CONFIG_DIR"] = str(config_dir(account))
    cmd = ["kaggle", *args]
    if dry:
        print(
            "   [dry-run]",
            " ".join(cmd),
            f"(KAGGLE_CONFIG_DIR={env['KAGGLE_CONFIG_DIR']})",
        )
        return ""
    result = subprocess.run(
        cmd, env=env, text=True, capture_output=capture, check=False
    )
    if capture:
        return (result.stdout or "") + (result.stderr or "")
    return ""


def load_ledger() -> dict:
    if not LEDGER.exists():
        return {"dataset": {}, "runs": {}}
    try:
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"ledger at {LEDGER} is unreadable ({exc}); delete it to start fresh"
        ) from exc


def save_ledger(ledger: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def stage_dataset() -> Path:
    """Collect code + base model + prepared corpus into one uploadable dir."""
    stage = WORK / "dataset"
    try:
        if stage.exists():
            shutil.rmtree(stage)
        (stage / "training").mkdir(parents=True)
        for name in (
            "prepare_data.py",
            "finetune_marian.py",
            "lanes.py",
            "__init__.py",
        ):
            src = REPO / "training" / name
            if src.exists():
                shutil.copy2(src, stage / "training" / name)
    except OSError as exc:
        raise SystemExit(f"cannot stage the dataset under {stage}: {exc}") from exc
    model_src = REPO / "models" / "opus-mt-vi-en-src"
    if not model_src.exists():
        raise SystemExit(f"missing base model at {model_src}; run `make models`")
    shutil.copytree(model_src, stage / "opus-mt-vi-en-src")
    train_dir = REPO / "eval_data" / "train"
    for name in (
        "train_mix.tsv",
        "train_general.tsv",
        "train_register.tsv",
        "dev.tsv",
        "manifest.json",
    ):
        src = train_dir / name
        if not src.exists():
            raise SystemExit(f"missing {src}; run `python -m training.prepare_data`")
        shutil.copy2(src, stage / name)
    meta = {
        "title": DATASET_TITLE,
        "id": None,  # filled per account at upload
        "licenses": [{"name": "CC0-1.0"}, {"name": "CC-BY-2.0"}],
        "description": (
            "Offline VI-EN MT fine-tune inputs: Opus-MT vi-en base checkpoint "
            "(CC BY 4.0), ParaCrawl-Bonus vi-en (CC0), Tatoeba vi-en (CC BY 2.0 FR), "
            "KDE4 strings (GPL text). No organizer corpus is included."
        ),
    }
    (stage / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return stage


def train_argv(lane: str) -> list[str]:
    """The trainer arguments for one lane, as a list (no shell quoting)."""
    cfg = LANES[lane]
    argv = [
        "--train",
        cfg.get("train", "train_mix.tsv"),
        "--dev",
        "dev.tsv",
        "--model-dir",
        "./opus-mt-vi-en-src",
        "--out",
        "/kaggle/working/ckpt",
        "--max-hours",
        str(cfg.get("max_hours", 10)),
    ]
    for key, flag in LANE_ARG.items():
        if key in cfg:
            argv += [flag, str(cfg[key])]
    return argv


def notebook_for(lane: str, account: dict) -> str:
    """Build the kernel notebook for one lane.

    The argv list is embedded as JSON and run with ``sys.executable`` *inside* the
    kernel. Interpolating the interpreter here would bake this machine's venv path
    into the notebook, and a shell string would break on Windows path separators.
    """
    cfg = LANES[lane]
    argv_literal = json.dumps(json.dumps(train_argv(lane)))
    lines = [
        "import json, os, subprocess, sys, time",
        "t0 = time.time()",
        "DS = '/kaggle/input/" + DATASET_SLUG + "'",
        "os.chdir(DS)",
        "sys.path.insert(0, DS)",
        "ARGS = json.loads(" + argv_literal + ")",
        'cmd = [sys.executable, "-m", "training.finetune_marian"] + ARGS',
        'print(" ".join(cmd), flush=True)',
        "rc = subprocess.call(cmd, cwd=DS)",
        'print("exit %s after %.1f min" % (rc, (time.time() - t0) / 60), flush=True)',
    ]
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# lane `" + lane + "`\n",
                cfg.get("note", "") + "\n",
                "\nAccount: "
                + account["name"]
                + ". Headless run: **Save & Run All**.\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in lines],
        },
    ]
    return json.dumps(
        {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        indent=1,
    )


def cmd_check(args: argparse.Namespace) -> int:
    accounts = read_accounts()
    plan = schedule()
    total = 0
    print(f"{'account':8s} {'user':20s} lanes")
    for acct in ACCOUNTS:
        total += len(plan[acct])
        print(
            f"{acct:8s} {accounts[acct]['username'][:20]:20s} {', '.join(plan[acct])}"
        )
    print(
        f"\n{len(ACCOUNTS)} accounts, {total} lanes = {len(ACCOUNTS) * 2} T4s if each session takes 2xT4"
    )
    print("quota: 30 GPU-h/week/account. Whether a 2xT4 hour costs 1 or 2 quota-hours")
    print(
        "       is unresolved - measure it with `launch.py probe` before planning the week."
    )
    if total != len(LANES):
        print(f"note: {len(LANES)} lanes defined, {total} scheduled", file=sys.stderr)
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    accounts = read_accounts()
    stage = stage_dataset()
    size = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file()) / 2**20
    print(f"staged {stage}  ({size:.0f} MB)")
    ledger = load_ledger()
    targets = ACCOUNTS if args.account == "all" else (args.account,)
    for acct in targets:
        meta_path = stage / "dataset-metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{meta_path} is not valid JSON: {exc}") from exc
        meta["id"] = accounts[acct]["username"] + "/" + DATASET_SLUG
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        existing = ledger["dataset"].get(acct)
        suffix = " (new version)" if existing else ""
        print(f"{acct}: dataset {meta['id']}{suffix}")
        if existing:
            kaggle(
                accounts[acct],
                "datasets",
                "version",
                "-p",
                str(stage),
                "-m",
                "lane sweep inputs",
                dry=args.dry_run,
            )
        else:
            kaggle(
                accounts[acct],
                "datasets",
                "create",
                "-p",
                str(stage),
                "--dir-mode",
                "zip",
                dry=args.dry_run,
            )
        ledger["dataset"][acct] = meta["id"]
    save_ledger(ledger)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    accounts = read_accounts()
    plan = schedule()
    ledger = load_ledger()
    targets: list[tuple[str, str]] = []
    if args.lane:
        pool = ACCOUNTS if args.account == "all" else (args.account,)
        targets = [(acct, args.lane) for acct in pool]
    else:
        for acct in ACCOUNTS:
            lanes = plan[acct]
            if args.wave - 1 < len(lanes):
                targets.append((acct, lanes[args.wave - 1]))

    for acct, lane in targets:
        if lane not in LANES:
            raise SystemExit(f"unknown lane {lane}; known: {', '.join(LANES)}")
        kdir = WORK / "kernels" / lane
        kdir.mkdir(parents=True, exist_ok=True)
        slug = ("kavi-mt-" + lane).replace("_", "-")
        (kdir / (slug + ".ipynb")).write_text(
            notebook_for(lane, accounts[acct]), encoding="utf-8"
        )
        meta = {
            "id": accounts[acct]["username"] + "/" + slug,
            "title": "kavi mt " + lane,
            "code_file": slug + ".ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": False,
            "dataset_sources": [
                ledger["dataset"].get(
                    acct, accounts[acct]["username"] + "/" + DATASET_SLUG
                )
            ],
            "kernel_sources": [],
            "competition_sources": [],
        }
        (kdir / "kernel-metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(f"{acct}: pushing {lane} -> {meta['id']}")
        print("      select the T4 x2 accelerator for this notebook in the Kaggle UI")
        kaggle(accounts[acct], "kernels", "push", "-p", str(kdir), dry=args.dry_run)
        ledger["runs"][lane] = {"account": acct, "ref": meta["id"]}
    save_ledger(ledger)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    accounts = read_accounts()
    ledger = load_ledger()
    if not ledger["runs"]:
        print("nothing pushed yet")
        return 0
    for lane, info in ledger["runs"].items():
        acct = info["account"]
        out = kaggle(
            accounts[acct],
            "kernels",
            "status",
            info["ref"],
            capture=True,
            dry=args.dry_run,
        )
        last = out.strip().splitlines()[-1] if out.strip() else "(dry-run)"
        print(f"{lane:18s} {acct:4s} {last}")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    accounts = read_accounts()
    ledger = load_ledger()
    known = ledger["runs"]
    lanes = [args.lane] if args.lane else list(known)
    for lane in lanes:
        if lane not in known:
            raise SystemExit(
                f"{lane} is not in the ledger; known runs: {', '.join(known) or 'none'}"
            )
        info = known[lane]
        dest = WORK / "output" / lane
        dest.mkdir(parents=True, exist_ok=True)
        kaggle(
            accounts[info["account"]],
            "kernels",
            "output",
            info["ref"],
            "-p",
            str(dest),
            dry=args.dry_run,
        )
        print(f"{lane}: pulled to {dest}")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Measure whether a 2xT4 hour costs one quota-hour or two.

    Kaggle documents "30 GPU hours per week" without settling it, and the account
    page counter is the only authority. This pushes a short GPU kernel that stamps
    the wall clock, so the counter delta over a known duration answers it - which
    decides whether the week holds ~244 or ~488 T4-hours.
    """
    accounts = read_accounts()
    acct = args.account
    kdir = WORK / "kernels" / "probe"
    kdir.mkdir(parents=True, exist_ok=True)
    slug = "kavi-mt-quota-probe"
    lines = [
        "import subprocess, time",
        "print('start', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)",
        "print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout)",
        "time.sleep(" + str(args.minutes * 60) + ")",
        "print('end', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)",
    ]
    cells = [
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in lines],
        }
    ]
    (kdir / (slug + ".ipynb")).write_text(
        json.dumps(
            {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5},
            indent=1,
        ),
        encoding="utf-8",
    )
    meta = {
        "id": accounts[acct]["username"] + "/" + slug,
        "title": "kavi mt quota probe",
        "code_file": slug + ".ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(
        f"probe: {args.minutes} min on {acct}. Select the T4 x2 accelerator in the UI first."
    )
    print(
        "Read the GPU-hours counter before and after; the delta over the runtime decides"
    )
    print("whether the week holds ~244 or ~488 T4-hours.")
    kaggle(accounts[acct], "kernels", "push", "-p", str(kdir), dry=args.dry_run)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # A parent parser so `--dry-run` works after the subcommand, which is what a
    # person types; the top-level copy keeps `--dry-run run` working too.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dry-run", action="store_true", help="stage and print, do not call the API"
    )
    ap.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)

    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", parents=[common]).set_defaults(fn=cmd_check)

    dataset = sub.add_parser("dataset", parents=[common])
    dataset.add_argument("--account", default="all")
    dataset.set_defaults(fn=cmd_dataset)

    run = sub.add_parser("run", parents=[common])
    run.add_argument(
        "--wave", type=int, default=1, help="1 = first lane per account, 2 = second"
    )
    run.add_argument(
        "--lane", default=None, help="push one named lane instead of a wave"
    )
    run.add_argument("--account", default="all")
    run.set_defaults(fn=cmd_run)

    sub.add_parser("status", parents=[common]).set_defaults(fn=cmd_status)

    pull = sub.add_parser("pull", parents=[common])
    pull.add_argument("--lane", default=None)
    pull.set_defaults(fn=cmd_pull)

    probe = sub.add_parser("probe", parents=[common])
    probe.add_argument("--account", default="m1")
    probe.add_argument("--minutes", type=float, default=15)
    probe.set_defaults(fn=cmd_probe)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
