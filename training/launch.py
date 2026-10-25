"""Orchestrate the fine-tune lanes across Kaggle accounts.

Kaggle has no cross-account scheduling API, so this drives the `kaggle` CLI once
per account with that account's own credentials, and keeps a local ledger of what
was pushed where. Nothing here holds a token: credentials live in
``training/accounts.toml`` (gitignored) and are written to a per-account
``KAGGLE_CONFIG_DIR`` at run time, never into a kernel and never into git.

Accounts are whatever that file holds, in file order — adding a teammate's token
re-balances the sweep without renaming anything. Entries still holding a
placeholder are skipped, so the lanes run on one account, two, or four.

    python -m training.launch check --verify     # parse + authenticate + print the plan
    python -m training.launch dataset            # stage + upload the inputs (per active account)
    python -m training.launch run --wave 1       # first lane on each active account
    python -m training.launch run --wave 2
    python -m training.launch status
    python -m training.launch pull --lane mix-lenp05
    python -m training.launch probe --minutes 15 # measure the quota counter

`--dry-run` works before or after the subcommand: it stages the files and prints
the CLI calls without executing them, which is how the notebook layout is reviewed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from .lanes import LANES, schedule

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "eval_data" / "kaggle"
LEDGER = WORK / "ledger.json"
DATASET_TITLE = "kavi mt finetune vi-en"
DATASET_SLUG = DATASET_TITLE.replace(" ", "-")
PLACEHOLDER_MARKS = ("REPLACE", "dummy", "REPLACE_ME")

# Lane config key -> training CLI flag. Anything not listed is not passed, so a
# lane cannot silently set a flag the trainer does not have.
# `kaggle kernels push --accelerator` sets request.machine_shape, so the GPU shape
# can be chosen at push time instead of in the UI. Kaggle's notebook offering for
# this class is 2xT4, which is one machine shape, not "one GPU".
ACCELERATOR = "NvidiaTeslaT4"

LANE_ARG = {
    "lr": "--lr",
    "epochs": "--epochs",
    "length_penalty": "--length-penalty",
    "batch_size": "--batch-size",
    "max_steps": "--max-steps",
}


def mask(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 6:
        return value[:2] + "*" * (len(value) - 2)
    return value[:3] + "*" * (len(value) - 6) + value[-3:]


def read_accounts() -> list[dict]:
    """Parse training/accounts.toml into file order, flagging placeholders."""
    path = REPO / "training" / "accounts.toml"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\ncopy training/accounts.toml.example and fill in the\n"
            "Kaggle username/key pairs (Kaggle -> Account -> Create New Token)."
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc

    entries: list[dict] = []
    current: dict = {}
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            if current:
                entries.append(current)
            current = {}
            continue
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        current[key] = value.strip().strip('"').strip("'")
    if current:
        entries.append(current)

    for entry in entries:
        name = entry.get("name") or "(unnamed)"
        user = entry.get("username", "")
        key = entry.get("key", "")
        entry["name"] = name
        entry["placeholder"] = (
            # An empty value is the loudest placeholder there is, but it cannot
            # live in PLACEHOLDER_MARKS: every string starts with "".
            not user
            or not key
            or any(user.startswith(mark) for mark in PLACEHOLDER_MARKS)
            or any(key.startswith(mark) for mark in PLACEHOLDER_MARKS)
            or len(key) < 20
        )
    if not entries:
        raise SystemExit(f"{path} holds no [[accounts]] blocks")
    return entries


def select(accounts: list[dict], spec: str | None) -> list[dict]:
    """Resolve --accounts: 'all'/'active' = every real account, or a name list."""
    if spec in (None, "", "all", "active"):
        active = [a for a in accounts if not a["placeholder"]]
        if not active:
            raise SystemExit(
                "no usable account yet: every entry in training/accounts.toml still "
                "holds a placeholder token"
            )
        return active
    wanted = [part.strip() for part in spec.split(",") if part.strip()]
    chosen: list[dict] = []
    for name in wanted:
        match = next((a for a in accounts if a["name"] == name), None)
        if match is None:
            known = ", ".join(a["name"] for a in accounts)
            raise SystemExit(f"no account named {name!r}; accounts.toml holds: {known}")
        if match["placeholder"]:
            raise SystemExit(f"account {name!r} still holds a placeholder token")
        chosen.append(match)
    return chosen


def by_name(accounts: list[dict], name: str) -> dict:
    match = next((a for a in accounts if a["name"] == name), None)
    if match is None:
        raise SystemExit(
            f"ledger names account {name!r}, which accounts.toml no longer holds"
        )
    if match["placeholder"]:
        raise SystemExit(
            f"ledger names account {name!r}, whose token is now a placeholder"
        )
    return match


def account_env(account: dict) -> dict:
    """Environment carrying exactly one account's credentials.

    Kaggle's newer tokens (``KGAT_...``) are a single API token, not the legacy
    username+long-key pair, so they go in as ``KAGGLE_API_TOKEN``. The CLI applies
    ``KAGGLE_*`` environment values after kaggle.json, so a stale global credential
    cannot override this - and a subprocess gets exactly one identity.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("KAGGLE_")}
    env["KAGGLE_API_TOKEN"] = account["key"]
    env["KAGGLE_USERNAME"] = account["username"]
    return env


def run_cli(
    account: dict, *args: str, dry: bool = False
) -> subprocess.CompletedProcess:
    cmd = ["kaggle", *args]
    if dry:
        print("   [dry-run]", " ".join(cmd), f"(token for {account['name']})")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(
        cmd, env=account_env(account), text=True, capture_output=True, check=False
    )


def kaggle(account: dict, *args: str, dry: bool = False, capture: bool = False) -> str:
    result = run_cli(account, *args, dry=dry)
    if capture:
        return (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        tail = ((result.stdout or "") + (result.stderr or "")).strip().splitlines()[-1:]
        print(f"   kaggle exited {result.returncode}: {' '.join(tail)}")
    return ""


def check_auth(account: dict) -> tuple[bool, str, dict | None]:
    """One authenticated call that also reports quota.

    Judged on the exit code, not on scanning the message: an unauthenticated run
    prints "Authentication required", which contains none of the obvious markers
    (401/403/unauthorized) and silently passed an earlier string-matching version.
    """
    if account["placeholder"]:
        return False, "placeholder token", None
    result = run_cli(account, "quota", "--format", "json")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        first = detail[0] if detail else f"exit {result.returncode}"
        return False, first[:90], None
    payload = (result.stdout or "").strip()
    start = payload.find("[")
    try:
        rows = json.loads(payload[start:]) if start >= 0 else []
    except json.JSONDecodeError:
        return True, "authenticated (quota unparsed)", None
    quota = {row.get("resource", "?"): row for row in rows if isinstance(row, dict)}
    gpu = quota.get("GPU", {})
    summary = f"GPU {gpu.get('used', '?')} used / {gpu.get('remaining', '?')} left of {gpu.get('total', '?')}"
    return True, summary, gpu or None


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
    print(f"{'account':10s} {'username':16s} {'token':20s} state")
    for acct in accounts:
        if acct["placeholder"]:
            print(
                f"{acct['name']:10s} {mask(acct['username']):16s} {mask(acct['key']):20s} "
                "placeholder - waiting for owner"
            )
            continue
        ok, detail, _ = (
            check_auth(acct)
            if args.verify
            else (True, "token present (not verified)", None)
        )
        state = ("OK" if ok else "FAIL") + ": " + detail
        print(
            f"{acct['name']:10s} {mask(acct['username']):16s} {mask(acct['key']):20s} {state}"
        )

    active = select(accounts, args.accounts)
    plan = schedule([a["name"] for a in active])
    print(f"\n{len(active)} usable account(s) of {len(accounts)}; lanes per account:")
    for name, lanes in plan.items():
        print(f"  {name:10s} {len(lanes)} lanes: {', '.join(lanes)}")
    covered = sum(1 for lanes in plan.values() for _ in lanes)
    print(f"\n{covered} of {len(LANES)} lanes covered by the current accounts.")
    if covered < len(LANES):
        print(f"{len(LANES) - covered} lane(s) wait for the remaining tokens; re-run")
        print("`check` after each arrives - the split re-balances automatically.")
    print(
        "\nquota: 30 GPU-h/week/account. Whether a 2xT4 hour costs 1 or 2 quota-hours"
    )
    print(
        "       is unresolved - measure it with `launch.py probe` before planning the week."
    )
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    accounts = read_accounts()
    targets = select(accounts, args.accounts)
    stage = stage_dataset()
    size = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file()) / 2**20
    print(f"staged {stage}  ({size:.0f} MB) -> {len(targets)} account(s)")
    ledger = load_ledger()
    for acct in targets:
        meta_path = stage / "dataset-metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{meta_path} is not valid JSON: {exc}") from exc
        meta["id"] = acct["username"] + "/" + DATASET_SLUG
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        existing = ledger["dataset"].get(acct["name"])
        suffix = " (new version)" if existing else ""
        print(f"{acct['name']}: dataset {meta['id']}{suffix}")
        if existing:
            kaggle(
                acct,
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
                acct,
                "datasets",
                "create",
                "-p",
                str(stage),
                "--dir-mode",
                "zip",
                dry=args.dry_run,
            )
        ledger["dataset"][acct["name"]] = meta["id"]
    save_ledger(ledger)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    accounts = read_accounts()
    targets_pool = select(accounts, args.accounts)
    ledger = load_ledger()
    plan = schedule([a["name"] for a in targets_pool])
    targets: list[tuple[dict, str]] = []
    for acct in targets_pool:
        lanes = plan[acct["name"]]
        if args.lane:
            if args.lane not in LANES:
                raise SystemExit(f"unknown lane {args.lane}; known: {', '.join(LANES)}")
            targets.append((acct, args.lane))
        elif 0 <= args.wave - 1 < len(lanes):
            targets.append((acct, lanes[args.wave - 1]))
    if not targets:
        raise SystemExit(
            f"wave {args.wave} is past the end of the schedule for these accounts"
        )

    for acct, lane in targets:
        kdir = WORK / "kernels" / lane
        kdir.mkdir(parents=True, exist_ok=True)
        slug = ("kavi-mt-" + lane).replace("_", "-")
        (kdir / (slug + ".ipynb")).write_text(
            notebook_for(lane, acct), encoding="utf-8"
        )
        meta = {
            "id": acct["username"] + "/" + slug,
            "title": "kavi mt " + lane,
            "code_file": slug + ".ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "machine_shape": ACCELERATOR,
            "enable_internet": False,
            "dataset_sources": [
                ledger["dataset"].get(
                    acct["name"], acct["username"] + "/" + DATASET_SLUG
                )
            ],
            "kernel_sources": [],
            "competition_sources": [],
        }
        (kdir / "kernel-metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(
            f"{acct['name']}: pushing {lane} -> {meta['id']} (machine_shape={ACCELERATOR})"
        )
        kaggle(acct, "kernels", "push", "-p", str(kdir), dry=args.dry_run)
        ledger["runs"][lane] = {"account": acct["name"], "ref": meta["id"]}
    save_ledger(ledger)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    accounts = read_accounts()
    ledger = load_ledger()
    if not ledger["runs"]:
        print("nothing pushed yet")
        return 0
    for lane, info in ledger["runs"].items():
        acct = by_name(accounts, info["account"])
        out = kaggle(
            acct, "kernels", "status", info["ref"], capture=True, dry=args.dry_run
        )
        last = out.strip().splitlines()[-1] if out.strip() else "(dry-run)"
        print(f"{lane:18s} {info['account']:10s} {last}")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    accounts = read_accounts()
    ledger = load_ledger()
    known = ledger["runs"]
    lanes = [args.lane] if args.lane else list(known)
    for lane in lanes:
        if lane not in known:
            listed = ", ".join(known) or "none"
            raise SystemExit(f"{lane} is not in the ledger; known runs: {listed}")
        info = known[lane]
        acct = by_name(accounts, info["account"])
        dest = WORK / "output" / lane
        dest.mkdir(parents=True, exist_ok=True)
        kaggle(
            acct, "kernels", "output", info["ref"], "-p", str(dest), dry=args.dry_run
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
    acct = select(accounts, args.accounts)[0]
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
        "id": acct["username"] + "/" + slug,
        "title": "kavi mt quota probe",
        "code_file": slug + ".ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": ACCELERATOR,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"probe: {args.minutes} min on {acct['name']} (machine_shape={ACCELERATOR}).")
    print(
        "Read the GPU-hours counter before and after; the delta over the runtime decides"
    )
    print("whether the week holds ~244 or ~488 T4-hours.")
    kaggle(acct, "kernels", "push", "-p", str(kdir), dry=args.dry_run)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # A parent parser so `--dry-run` works after the subcommand, which is what a
    # person types; the top-level copy keeps `--dry-run run` working too.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dry-run", action="store_true", help="stage and print, do not call the API"
    )
    common.add_argument(
        "--accounts",
        default="all",
        help="comma-separated account names from accounts.toml, or 'all' (default: every real token)",
    )
    ap.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)

    sub = ap.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check", parents=[common])
    check.add_argument(
        "--verify", action="store_true", help="make one authenticated call per account"
    )
    check.set_defaults(fn=cmd_check)

    dataset = sub.add_parser("dataset", parents=[common])
    dataset.set_defaults(fn=cmd_dataset)

    run = sub.add_parser("run", parents=[common])
    run.add_argument(
        "--wave", type=int, default=1, help="1 = first lane per account, 2 = second"
    )
    run.add_argument(
        "--lane", default=None, help="push one named lane instead of a wave"
    )
    run.set_defaults(fn=cmd_run)

    sub.add_parser("status", parents=[common]).set_defaults(fn=cmd_status)

    pull = sub.add_parser("pull", parents=[common])
    pull.add_argument("--lane", default=None)
    pull.set_defaults(fn=cmd_pull)

    probe = sub.add_parser("probe", parents=[common])
    probe.add_argument("--minutes", type=float, default=15)
    probe.set_defaults(fn=cmd_probe)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
