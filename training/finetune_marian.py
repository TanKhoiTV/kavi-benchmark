"""Fine-tune Helsinki-NLP/opus-mt-vi-en on the prepared VI->EN corpus.

Kaggle-side. Plain PyTorch rather than ``Seq2SeqTrainer``: the training image has
no ``accelerate``/``evaluate``, the loop needs a wall-clock budget that stops
cleanly before the 12 h session cap, and checkpoint/resume has to survive being
split across sessions. Those three are why this is hand-written.

Single GPU or 2xT4 data-parallel on one account (no NVLink on Kaggle, so DDP over
PCIe; a 77M model is small enough that this is fine):

    python -m training.finetune_marian --train train_mix.tsv --dev dev.tsv \
        --model-dir opus-mt-vi-en-src --out ckpt --lr 3e-5 --max-hours 8
    torchrun --nproc_per_node=2 -m training.finetune_marian ...

Resume is the default when the output directory already holds a checkpoint:
a 30-hour job is three sessions, not one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class PairDataset(Dataset):
    """TSV of ``<vi>\\t<en>``, tokenized on the fly.

    90k pairs is ~50 MB in memory as text, so the file is read once and the
    tokenizer runs per item in the worker processes. Pre-tokenizing would save
    time only if the run were longer than the tokenization, which it is not.
    """

    def __init__(self, path: str | Path, tokenizer, max_src: int, max_tgt: int) -> None:
        self.rows: list[tuple[str, str]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2 and parts[0] and parts[1]:
                    self.rows.append((parts[0], parts[1]))
        if not self.rows:
            raise SystemExit(f"no usable rows in {path}")
        self.tok = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        vi, en = self.rows[i]
        src = self.tok(vi, max_length=self.max_src, truncation=True, padding=False)[
            "input_ids"
        ]
        tgt = self.tok(en, max_length=self.max_tgt, truncation=True, padding=False)[
            "input_ids"
        ]
        return {"input_ids": torch.tensor(src), "labels": torch.tensor(tgt)}


class Collator:
    """Right-pad source ids; pad labels with -100 so padding is not a target."""

    def __init__(self, pad_id: int) -> None:
        self.pad = pad_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad = self.pad
        max_src = max(int(b["input_ids"].numel()) for b in batch)
        max_tgt = max(int(b["labels"].numel()) for b in batch)
        input_ids = torch.full((len(batch), max_src), pad, dtype=torch.long)
        attention = torch.zeros((len(batch), max_src), dtype=torch.long)
        labels = torch.full((len(batch), max_tgt), -100, dtype=torch.long)
        for i, item in enumerate(batch):
            n = int(item["input_ids"].numel())
            input_ids[i, :n] = item["input_ids"]
            attention[i, :n] = 1
            m = int(item["labels"].numel())
            labels[i, :m] = item["labels"]
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def distributed() -> tuple[bool, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 1
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])


def latest_checkpoint(out: Path) -> Path | None:
    cands = sorted(
        (p for p in out.glob("step-*") if (p / "config.json").exists()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    return cands[-1] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_mix.tsv")
    ap.add_argument("--dev", default="dev.tsv")
    ap.add_argument("--model-dir", default="opus-mt-vi-en-src")
    ap.add_argument("--model-id", default=None, help="HF hub id instead of --model-dir")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weights", default="adamw")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = derive from epochs")
    ap.add_argument("--warmup-frac", type=float, default=0.06)
    ap.add_argument("--max-src", type=int, default=128)
    ap.add_argument("--max-tgt", type=int, default=128)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument(
        "--length-penalty",
        type=float,
        default=0.0,
        help="upsample-weight short targets: weight=(median_len/len)^alpha, clipped [0.5,2]; 0 = off",
    )
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument(
        "--max-hours", type=float, default=10.0, help="stop before the session cap"
    )
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--smoke", action="store_true", help="20 steps on 200 pairs, CPU-ok"
    )
    args = ap.parse_args()

    rank, world = 0, 1
    ddp, rank, world = distributed()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)

    from transformers import MarianMTModel, MarianTokenizer

    source = args.model_id or args.model_dir
    tokenizer = MarianTokenizer.from_pretrained(source)
    # `core` stays the HF model: it owns save_pretrained and the parameters DDP
    # mutates in place, so saving it is equivalent to unwrapping `net.module`
    # and avoids the union type that made every later call site ambiguous.
    core = MarianMTModel.from_pretrained(source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = core.to(device)  # type: ignore[assignment]
    net: torch.nn.Module = core

    train_ds = PairDataset(args.train, tokenizer, args.max_src, args.max_tgt)
    dev_ds = PairDataset(args.dev, tokenizer, args.max_src, args.max_tgt)
    if args.smoke:
        train_ds.rows, dev_ds.rows = train_ds.rows[:200], dev_ds.rows[:20]
        args.max_steps, args.batch_size, args.save_every, args.eval_every = (
            20,
            8,
            10,
            10,
        )

    collate = Collator(tokenizer.pad_token_id)
    sampler = (
        DistributedSampler(train_ds, num_replicas=world, rank=rank) if ddp else None
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=2,
        collate_fn=collate,
        drop_last=ddp,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, collate_fn=collate)
    if ddp:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[rank] if device.type == "cuda" else None
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = max(1, math.ceil(len(loader) / max(1, args.grad_accum)))
    total = args.max_steps or int(steps_per_epoch * args.epochs)
    warmup = int(total * args.warmup_frac)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")  # pyright: ignore[reportPrivateImportUsage]

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return args.lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

    start_step = 0
    if not args.no_resume and not args.smoke:
        ck = latest_checkpoint(out)
        if ck is not None:
            if is_main:
                print(f"resuming from {ck}", flush=True)
            core.load_state_dict(
                torch.load(
                    ck / "pytorch_model.bin", map_location="cpu", weights_only=True
                ),
                strict=False,
            )
            opt_path = ck / "optimizer.pt"
            if opt_path.exists():
                opt.load_state_dict(
                    torch.load(opt_path, map_location="cpu", weights_only=False)
                )
            start_step = int(
                (ck / "status.json").read_text().split('"step": ')[1].split(",")[0]
            )

    # Length penalty, if asked for, weights short targets up so the model spends
    # more of its capacity on the brevity the shipped configuration needs.
    tgt_lengths = [len(t.split()) for _, t in train_ds.rows]
    median_len = sorted(tgt_lengths)[len(tgt_lengths) // 2]

    def weights_for(labels: torch.Tensor) -> torch.Tensor:
        if args.length_penalty <= 0:
            return torch.ones(labels.shape[0], device=labels.device)
        lens = (labels != -100).sum(dim=1).clamp(min=1).float()
        w = (median_len / lens) ** args.length_penalty
        return w.clamp(0.5, 2.0)

    if is_main:
        print(
            json.dumps(
                {
                    "pairs": len(train_ds),
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total,
                    "batch": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "world": world,
                    "device": str(device),
                    "lr": args.lr,
                    "length_penalty": args.length_penalty,
                }
            ),
            flush=True,
        )

    t0 = time.time()
    step = start_step
    net.train()
    done = step >= total
    while not done:
        if sampler is not None:
            sampler.set_epoch(step // max(1, steps_per_epoch))
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):  # pyright: ignore[reportPrivateImportUsage]
                logits = net(**batch).logits
                loss_fct = torch.nn.CrossEntropyLoss(
                    label_smoothing=args.label_smoothing,
                    reduction="none",
                    ignore_index=-100,
                )
                token_loss = loss_fct(
                    logits.view(-1, logits.size(-1)), batch["labels"].view(-1)
                )
                token_loss = token_loss.view(batch["labels"].shape[0], -1).sum(dim=1)
                n_tok = (batch["labels"] != -100).sum(dim=1).clamp(min=1)
                sample_loss = token_loss / n_tok
                loss = (
                    sample_loss * weights_for(batch["labels"])
                ).mean() / args.grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                for group in opt.param_groups:
                    group["lr"] = lr_at(step)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1

                if is_main and step % 25 == 0:
                    elapsed = (time.time() - t0) / 60
                    print(
                        f"step {step}/{total}  loss {loss.item() * args.grad_accum:.4f}  "
                        f"lr {opt.param_groups[0]['lr']:.2e}  {elapsed:.1f} min",
                        flush=True,
                    )
                if step % args.eval_every == 0:
                    net.eval()
                    tot, n = 0.0, 0
                    with torch.no_grad():
                        for db in dev_loader:
                            db = {k: v.to(device) for k, v in db.items()}
                            tot += float(net(**db).loss)
                            n += 1
                    if is_main:
                        print(
                            f"  dev loss {tot / max(1, n):.4f} at step {step}",
                            flush=True,
                        )
                    net.train()
                if step % args.save_every == 0 and is_main:
                    ck = out / f"step-{step}"
                    ck.mkdir(parents=True, exist_ok=True)
                    core.save_pretrained(ck)
                    tokenizer.save_pretrained(ck)
                    torch.save(opt.state_dict(), ck / "optimizer.pt")
                    (ck / "status.json").write_text(
                        json.dumps({"step": step, "total": total})
                    )
                    print(f"  saved {ck}", flush=True)
                if step >= total:
                    done = True
                    break
                if (time.time() - t0) / 3600 > args.max_hours:
                    if is_main:
                        ck = out / f"step-{step}"
                        ck.mkdir(parents=True, exist_ok=True)
                        core.save_pretrained(ck)
                        tokenizer.save_pretrained(ck)
                        torch.save(opt.state_dict(), ck / "optimizer.pt")
                        (ck / "status.json").write_text(
                            json.dumps(
                                {"step": step, "total": total, "stop": "time_budget"}
                            )
                        )
                        print(
                            f"time budget reached at step {step}; saved {ck}",
                            flush=True,
                        )
                    done = True
                    break

    if is_main:
        final = out / "final"
        final.mkdir(parents=True, exist_ok=True)
        core.save_pretrained(final)
        tokenizer.save_pretrained(final)
        (final / "status.json").write_text(
            json.dumps(
                {"step": step, "total": total, "minutes": (time.time() - t0) / 60}
            )
        )
        print(f"done at step {step}/{total}; wrote {final}", flush=True)
    if ddp:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
