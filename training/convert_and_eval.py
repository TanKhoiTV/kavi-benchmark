"""Evaluate a fine-tuned checkpoint on the held-out domain pairs, locally.

Deliberately host-side: the organizer corpus is not redistributable, so it never
goes to Kaggle. Training runs there, scoring runs here, and the only thing that
crosses the boundary is the checkpoint.

Reports the same four things the ship gate in
``docs/reference/mt-finetune-data-plan.md`` §5 asks for, plus a paired bootstrap
CI so a delta is not read off noise (the domain set is 100 items, and CIs there
run +-4 to 7 sentence BLEU):

  * sentence BLEU mean and corpus BLEU, both named
  * comma rate and terminal-mark rate  (gold: 42/100 and 100/100)
  * mean words per hypothesis          (gold: 8.46)
  * degenerate-loop count             (repeated 3-gram covering >=40% of output)

Usage:

    # pinned baseline, to confirm the harness reproduces 24.95 / 22.13
    python -m training.convert_and_eval --ct2 models/opus-mt-vi-en-ct2 --gold <xlsx> --beam 1

    # a lane's checkpoint: convert to int8 first, then score
    python -m training.convert_and_eval --ckpt eval_data/kaggle/output/mix-lenp05/ckpt/final \
        --convert --gold <xlsx> --beam 1 --baseline eval_data/train/base_greedy.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

# The pinned baseline on the same 100 pairs, from mt-finetune-data-plan §1.4.
PINNED = {
    "1": {
        "sentence_bleu_mean": 24.95,
        "corpus_bleu": 22.13,
        "comma_rate": 0.45,
        "words": 10.76,
    },
    "4": {
        "sentence_bleu_mean": 28.82,
        "corpus_bleu": 29.33,
        "comma_rate": 0.43,
        "words": 9.31,
    },
}
GOLD_WORDS = 8.46
GOLD_COMMAS = 0.42


def load_gold(path: Path) -> list[tuple[str, str, str]]:
    """Read the supplied pairs as (id, vi, en) from the organizers' spreadsheet."""
    import pandas as pd

    df = pd.read_excel(path)
    cols = list(df.columns)
    id_col = cols[0]
    vi_col = next(c for c in cols if "Việt" in c or "Viet" in c)
    en_col = next(c for c in cols if "Anh" in c or "English" in c)
    return [
        (str(r[id_col]), str(r[vi_col]).strip(), str(r[en_col]).strip())
        for _, r in df.iterrows()
        if str(r[vi_col]).strip() and str(r[en_col]).strip()
    ]


def is_degenerate(text: str) -> bool:
    words = text.split()
    if len(words) < 6:
        return False
    grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    top = max(grams.count(g) for g in set(grams))
    return top / len(grams) >= 0.40


def translate_ct2(
    model_dir: Path, src_spm: Path, tgt_spm: Path, pairs, beam: int
) -> list[str]:
    import ctranslate2
    import sentencepiece as spm

    translator = ctranslate2.Translator(
        str(model_dir), device="cpu", compute_type="int8"
    )
    sp_src, sp_tgt = spm.SentencePieceProcessor(), spm.SentencePieceProcessor()
    # sentencepiece ships no stubs for these C-extension methods; same convention
    # as bench/candidates/opusmt_mt.py.
    sp_src.load(str(src_spm))  # pyright: ignore[reportAttributeAccessIssue]
    sp_tgt.load(str(tgt_spm))  # pyright: ignore[reportAttributeAccessIssue]
    out = []
    for _, vi, _ in pairs:
        tokens = sp_src.encode(vi, out_type=str)  # pyright: ignore[reportAttributeAccessIssue]
        # Same guard as bench/decode.py: a flat cap let greedy loop.
        cap = min(256, max(48, int(len(tokens) * 1.5)))
        res = translator.translate_batch(
            [tokens],
            beam_size=beam,
            max_decoding_length=cap,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
        out.append(sp_tgt.decode(res[0].hypotheses[0]))  # pyright: ignore[reportAttributeAccessIssue]
    return out


def translate_hf(model_dir: Path, pairs, beam: int) -> list[str]:
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    tok = MarianTokenizer.from_pretrained(str(model_dir))
    model = MarianMTModel.from_pretrained(str(model_dir)).eval()
    out = []
    for _, vi, _ in pairs:
        batch = tok(vi, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            ids = model.generate(**batch, num_beams=beam, max_new_tokens=256)
        out.append(tok.decode(ids[0], skip_special_tokens=True))
    return out


def convert(ckpt: Path, dest: Path) -> None:
    from ctranslate2.converters import TransformersConverter

    dest.mkdir(parents=True, exist_ok=True)
    TransformersConverter(str(ckpt)).convert(str(dest), quantization="int8", force=True)
    # The tokenizers must travel with the weights: no network at run time.
    for name in ("source.spm", "target.spm", "vocab.json", "shared_vocabulary.json"):
        src = ckpt / name
        if src.exists():
            (dest / name).write_bytes(src.read_bytes())
    print(f"converted {ckpt} -> {dest}")


def bootstrap(diffs: list[float], iters: int = 10000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)
    )
    lo, hi = means[int(0.025 * iters)], means[int(0.975 * iters) - 1]
    return {
        "diff": round(sum(diffs) / n, 3),
        "ci95": [round(lo, 3), round(hi, 3)],
        "p_leq_zero": round(sum(1 for m in means if m <= 0) / iters, 4),
        "excludes_zero": not (lo <= 0 <= hi),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct2", default=None, help="existing CT2 model directory")
    ap.add_argument(
        "--ckpt",
        default=None,
        help="HF checkpoint dir (needs --convert or --backend hf)",
    )
    ap.add_argument(
        "--convert", action="store_true", help="convert --ckpt to int8 CT2 first"
    )
    ap.add_argument("--convert-out", default=None, help="where the CT2 model lands")
    ap.add_argument("--backend", choices=("ct2", "hf"), default="ct2")
    ap.add_argument(
        "--gold", required=True, help="organizers' xlsx (not redistributed)"
    )
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument(
        "--baseline", default=None, help="json of a previous run, for a paired CI"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--spm-dir", default="models/opus-mt-vi-en-src")
    args = ap.parse_args()

    pairs = load_gold(Path(args.gold))
    refs = [en for _, _, en in pairs]
    print(f"gold pairs: {len(pairs)}")

    if args.backend == "ct2":
        model_dir = Path(args.ct2) if args.ct2 else None
        if args.ckpt and (args.convert or model_dir is None):
            dest = Path(args.convert_out or f"{args.ckpt}-ct2-int8")
            if args.convert:
                convert(Path(args.ckpt), dest)
            model_dir = dest
        if model_dir is None:
            raise SystemExit("need --ct2 or --ckpt")
        spm_dir = Path(args.spm_dir)
        hyps = translate_ct2(
            model_dir, spm_dir / "source.spm", spm_dir / "target.spm", pairs, args.beam
        )
    else:
        if not args.ckpt:
            raise SystemExit("--backend hf needs --ckpt")
        hyps = translate_hf(Path(args.ckpt), pairs, args.beam)

    import sacrebleu

    sent = [
        sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyps, refs, strict=True)
    ]
    result = {
        "n": len(hyps),
        "beam": args.beam,
        "sentence_bleu_mean": round(st.mean(sent), 2),
        "corpus_bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        "comma_rate": round(sum(1 for h in hyps if "," in h) / len(hyps), 3),
        "terminal_rate": round(
            sum(1 for h in hyps if h.rstrip().endswith((".", "?", "!"))) / len(hyps), 3
        ),
        "words_mean": round(st.mean(len(h.split()) for h in hyps), 2),
        "degenerate": sum(1 for h in hyps if is_degenerate(h)),
        "hyps": hyps,
        "per_item_bleu": [round(x, 2) for x in sent],
    }
    result["words_vs_gold"] = round(result["words_mean"] / GOLD_WORDS, 3)

    pinned = PINNED.get(str(args.beam))
    if pinned:
        result["vs_pinned"] = {
            "sentence_bleu_mean": round(
                result["sentence_bleu_mean"] - pinned["sentence_bleu_mean"], 2
            ),
            "corpus_bleu": round(result["corpus_bleu"] - pinned["corpus_bleu"], 2),
            "comma_rate": round(result["comma_rate"] - pinned["comma_rate"], 3),
            "words": round(result["words_mean"] - pinned["words"], 2),
        }

    print(
        f"  sentence BLEU {result['sentence_bleu_mean']:.2f}  corpus BLEU {result['corpus_bleu']:.2f}"
        f"  commas {result['comma_rate']:.2f}  terminal {result['terminal_rate']:.2f}"
        f"  words {result['words_mean']:.2f} ({result['words_vs_gold']:.2f}x gold)"
        f"  degenerate {result['degenerate']}"
    )
    if pinned:
        print(f"  vs pinned beam={args.beam}: {result['vs_pinned']}")

    gate_pass = (
        result["comma_rate"] >= 0.40
        and result["terminal_rate"] >= 0.98
        and result["words_vs_gold"] <= 1.10
        and result["degenerate"] == 0
    )
    result["gate_pass_brevity_punctuation"] = gate_pass
    print(f"  brevity+punctuation gate: {'PASS' if gate_pass else 'FAIL'}")

    if args.baseline:
        base_path = Path(args.baseline)
        if not base_path.exists():
            raise SystemExit(
                f"no baseline run at {base_path}; score the pinned model first"
            )
        with base_path.open(encoding="utf-8") as fh:
            try:
                base = json.load(fh)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{base_path} is not valid JSON: {exc}") from exc
        if "per_item_bleu" not in base or len(base["per_item_bleu"]) != len(
            result["per_item_bleu"]
        ):
            raise SystemExit(
                f"{base_path} does not hold {len(result['per_item_bleu'])} per-item scores; "
                "a paired CI needs the same items in the same order"
            )
        diffs = [
            a - b
            for a, b in zip(result["per_item_bleu"], base["per_item_bleu"], strict=True)
        ]
        result["vs_baseline"] = bootstrap(diffs)
        b = result["vs_baseline"]
        verdict = "excludes zero" if b["excludes_zero"] else "INSIDE noise"
        print(f"  paired vs baseline: {b['diff']:+.2f}  95% CI {b['ci95']}  {verdict}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
