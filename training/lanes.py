"""The lane matrix: what each of the eight T4 lanes trains.

Eight T4s across four accounts is **four concurrent sessions of 2xT4**, not one
eight-way job — there is no network between accounts, so the parallelism has to
be across configurations. Two ways to spend it:

  * one job per account on 2xT4 (data-parallel DDP) — finishes each config
    sooner, explores fewer of them;
  * two independent jobs per account, one T4 each — explores twice as many
    configurations in the same wall clock.

`mode` below is the second. A 77M Marian model gets little from PCIe DDP (no
NVLink on Kaggle), and the open questions here are *which* configuration wins,
not how fast one finishes. Passing `--ddp` to the launcher switches an account
to the first mode if a lane turns out to need it.

Every lane trains the same prepared corpus (``train_mix.tsv``) unless it names a
different ``--train`` file; the sweep varies the things the plan says are open —
learning rate, schedule length, and the length penalty — not the data.
"""

from __future__ import annotations

LANES: dict[str, dict] = {
    # Two baseline lanes at the conventional Marian fine-tune rate.
    "mix-lr3e5-e3": {"lr": 3e-5, "epochs": 3, "note": "baseline: 90k mix, 3 epochs"},
    "mix-lr3e5-e1": {
        "lr": 3e-5,
        "epochs": 1,
        "note": "one epoch is often enough for domain shifts",
    },
    # Learning-rate bracket around the baseline.
    "mix-lr1e5-e3": {
        "lr": 1e-5,
        "epochs": 3,
        "note": "conservative: less forgetting of FLEURS",
    },
    "mix-lr5e5-e3": {
        "lr": 5e-5,
        "epochs": 3,
        "note": "aggressive: faster domain fit, regression risk",
    },
    # Schedule length at the baseline rate.
    "mix-lr3e5-e6": {
        "lr": 3e-5,
        "epochs": 6,
        "note": "longer: does more exposure help or overfit",
    },
    # The brevity objective, swept as the plan requires (after domain fit).
    "mix-lenp05": {
        "lr": 2e-5,
        "epochs": 3,
        "length_penalty": 0.5,
        "note": "mild brevity pressure",
    },
    "mix-lenp10": {
        "lr": 2e-5,
        "epochs": 3,
        "length_penalty": 1.0,
        "note": "strong brevity pressure",
    },
    # General-distribution-only control: does the register slice matter at all?
    "general-lr3e5-e3": {
        "lr": 3e-5,
        "epochs": 3,
        "train": "train_general.tsv",
        "note": "control: no register slice",
    },
}

# Deterministic assignment: two lanes per account, so one account's pair differs
# as much as possible (baseline plus a probe) rather than both being adjacent.
ACCOUNTS = ("m1", "m2", "m3", "m4")


def schedule() -> dict[str, list[str]]:
    """Account -> the lanes it runs, in order.

    Interleaved rather than sliced: with a contiguous split, one account gets
    both brevity lanes and another both learning-rate extremes, so a single
    account dying loses a whole comparison.
    """
    names = list(LANES)
    return {acct: names[i :: len(ACCOUNTS)] for i, acct in enumerate(ACCOUNTS)}


if __name__ == "__main__":
    for acct, lanes in schedule().items():
        print(f"{acct}: {', '.join(lanes)}")
