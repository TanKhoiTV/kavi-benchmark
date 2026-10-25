"""The lane matrix: what each of the fine-tune lanes trains.

Accounts × 2xT4 is *their* concurrent sessions, not one big job — there is no
network between accounts, so the parallelism has to be across configurations. Two
ways to spend it:

  * one job per account on 2xT4 (data-parallel DDP) — finishes each config
    sooner, explores fewer of them;
  * two or more independent jobs per account, one T4 each — explores more
    configurations in the same wall clock.

`launch.py` uses the second. A 77M Marian model gets little from PCIe DDP (no
NVLink on Kaggle), and the open questions here are *which* configuration wins,
not how fast one finishes. `--ddp` switches an account to the first shape if a
lane turns out to need it.

Account labels are not fixed here: lanes are distributed over whatever accounts
`training/accounts.toml` actually holds, in file order, so adding the third and
fourth teammate's token re-balances the sweep without renaming anything.

Every lane trains the same prepared corpus (``train_mix.tsv``) unless it names a
different ``--train`` file; the sweep varies the things the plan leaves open —
learning rate, schedule length, and the length penalty — not the data.
"""

from __future__ import annotations

from collections.abc import Sequence

LANES: dict[str, dict] = {
    # Two baseline lanes at the conventional Marian fine-tune rate.
    "mix-lr3e5-e3": {"lr": 3e-5, "epochs": 3, "note": "baseline: 90k mix, 3 epochs"},
    "mix-lr3e5-e1": {
        "lr": 3e-5,
        "epochs": 1,
        "note": "one epoch is often enough for domain shifts",
    },
    # The punctuation-contract lane. The corpus was rebuilt so every English
    # target carries a terminal mark and the source matches ASR (lowercase,
    # mostly unpunctuated): the first sweep trained on targets that were only
    # 71% punctuated, and the model reproduced that rate as 65/100 instead of
    # the gold's 100/100. One epoch at the conservative end of the adaptation
    # bracket is the decisive test of the contract, so it is not the sweep.
    "pc-mix-lr1e5-e1": {
        "lr": 1e-5,
        "epochs": 1,
        "note": "punctuation-contract mix, conservative LR, one epoch",
    },
    # The tokenizer fix. Every lane above encoded labels with spm_source, so they
    # all learned a corrupted target distribution. Same corpus and schedule as
    # pc-mix-lr1e5-e1, so the delta isolates the fix itself.
    "pcfix-lr1e5-e1": {
        "lr": 1e-5,
        "epochs": 1,
        "note": "text_target fix on the contract mix, one epoch",
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

# The nominal plan, used only for prose and tests: four accounts, two lanes each.
ACCOUNTS = ("m1", "m2", "m3", "m4")


def schedule(accounts: Sequence[str] | None = None) -> dict[str, list[str]]:
    """Account -> the lanes it runs, in order.

    Interleaved rather than sliced: with a contiguous split, one account gets
    both brevity lanes and another both learning-rate extremes, so a single
    account dying loses a whole comparison. Fewer accounts means more lanes each,
    still interleaved, so the highest-value lanes come first on every account.
    """
    names = list(LANES)
    accts = list(accounts) if accounts else list(ACCOUNTS)
    if not accts:
        return {}
    return {acct: names[i :: len(accts)] for i, acct in enumerate(accts)}


if __name__ == "__main__":
    import sys

    for account, lanes in schedule(sys.argv[1:] or None).items():
        print(f"{account}: {', '.join(lanes)}")
