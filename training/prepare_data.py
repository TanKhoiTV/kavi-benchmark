"""Build the VI->EN fine-tune corpus from licence-clean OPUS sources.

Implements tier A and the register-selected slice of
``docs/reference/mt-finetune-data-plan.md`` §2.2. Tier C (authored
service-encounter pairs) and tier D (factory vocabulary) are NOT produced here:
they are written by hand, because machine-assisted drafting imports exactly the
distribution the fine-tune exists to fix.

Sources, each with the licence string its OPUS package ships:

  ParaCrawl-Bonus  v9          CC0 1.0            the volume
  Tatoeba          v2026-07-08 CC BY 2.0 FR       short colloquial pairs
  KDE4             v2          GPL (as source)    UI-style imperatives

Blocked under ADR-029 and deliberately absent: CCMatrix/NLLB (no grant / NC),
OpenSubtitles (varies), CCAligned (research), QED (mixed), TED2020 and
NeuLab-TedTalks (CC BY-NC-ND), PhoMT (research-only), MTet (NC-SA), IWSLT15
(TED-derived NC-ND). WikiMatrix/wikimedia are CC BY-SA and belong to tier B,
which waits on the share-alike ruling; they are not read here.

Outputs, all under ``--out`` (gitignored, with ``eval_data/``):

  train_general.tsv   general-distribution pairs, length-bounded, deduplicated
  train_register.tsv  pairs that match the supplied corpus's measured register
  dev.tsv             held-out general pairs for overfit detection
  manifest.json       counts, licences, versions, filter parameters

Run:  python -m training.prepare_data
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
from pathlib import Path

# Measured on the organizers' supplied 100 VI->EN pairs (mt-finetune-data-plan §1.3):
# VI mean 8.16 words (median 7, max 20), EN mean 8.46 (median 8, max 25),
# 42/100 with a comma, 35/100 questions. The general pool is that distribution
# widened; the register slice is that distribution matched, because a slice whose
# mean is 14 words trains a model that writes 14-word sentences.
MIN_WORDS = 2
MAX_WORDS = 24
REG_MIN_WORDS = 3
REG_MAX_VI = 14
REG_MAX_EN = 18

# Punctuation contract (mt-finetune-data-plan §2.2). The first lane taught the
# model to stop punctuating because the corpus did: train_general carried a
# terminal mark on only 55.3% of its English rows against 100% in the supplied
# gold, and the fine-tune emitted terminal marks at 65% - tracking the corpus,
# not the references. Targets must be punctuated, and the source must look like
# the ASR output that feeds MT at run time: lowercase, usually unpunctuated.
TERMINAL = (".", "?", "!")

# Boilerplate and misalignment markers. Web crawl and software strings both
# survive a length filter looking like prose; these are what they look like.
BOILERPLATE = ("●", "•", "…", "©", "®", "™", "http", "www.", "&nbsp;")
ENUM = re.compile(r"^\s*(\d+[.)]|[a-z][.)])\s+", re.I)
TOO_NUMERIC = re.compile(r"(\d+[\d.,]*\s*){3,}")
# Speaker/section labels: "H:", "Q:", "A:", "MỘT:" survive cleaning and teach the
# model to emit a label token as the first target token.
LABEL = re.compile(r"^\s*(h|q|a|hỏi|đáp|một|m\d+)\s*:\s*", re.I)
# A pair whose English side is not English is an alignment failure, not data.
NOT_ENGLISH = ("¿", "¡", "«", "»", "Ã", "ñ", "ção")
EN_FUNCTION = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "and",
    "you",
    "i",
    "we",
    "it",
    "do",
    "does",
    "can",
    "could",
    "would",
    "will",
    "for",
    "in",
    "on",
    "at",
    "with",
    "my",
    "your",
    "this",
    "that",
    "have",
    "has",
    "not",
    "there",
    "what",
    "how",
    "when",
    "where",
}

# Service-encounter cues. These are the register the supplied pairs measure:
# hotel/travel, retail, office/event, parcel/logistics, clinic, directions.
VI_CUES = (
    "anh",
    "chị",
    "em",
    "ông",
    "bà",
    "tôi",
    "mình",
    "bạn",
    "vui lòng",
    "giúp",
    "xin",
    "cảm ơn",
    "được không",
    "bao nhiêu",
    "ở đâu",
    "khi nào",
    "mấy giờ",
    "phòng",
    "đơn",
    "hàng",
    "hoá đơn",
    "hóa đơn",
    "vé",
    "khách sạn",
    "đặt",
    "giao",
    "chuyển",
    "kho",
    "bưu phẩm",
    "hạn",
    "giá",
    "thanh toán",
    "địa chỉ",
    "hành lý",
    "chuyến",
    "bác sĩ",
    "thuốc",
    "bệnh viện",
    "cửa hàng",
    "nhà ga",
)
EN_CUES = (
    "could you",
    "can i",
    "can we",
    "do you",
    "does the",
    "how long",
    "how much",
    "where is",
    "where do",
    "which",
    "what time",
    "is there",
    "are there",
    "i'd like",
    "i would like",
    "please",
    "thank you",
    "sorry",
    "excuse me",
    "book",
    "booking",
    "reservation",
    "invoice",
    "receipt",
    "delivery",
    "deliver",
    "shipment",
    "warehouse",
    "parcel",
    "tracking",
    "warranty",
    "appointment",
    "doctor",
    "pharmacy",
    "hotel",
    "room",
    "ticket",
    "station",
    "address",
)

URL = re.compile(r"https?://\S+|www\.\S+")
TAGS = re.compile(r"<[^>]+>")
SPACES = re.compile(r"\s+")
# A Vietnamese line should carry at least one diacritic; a pair without any is
# usually untranslated source text or an alignment failure.
VI_DIACRITIC = re.compile(
    r"[ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]", re.I
)

SOURCES = (
    ("tatoeba", "Tatoeba", "v2026-07-08", "CC BY 2.0 FR"),
    ("kde4", "KDE4", "v2", "GPL (text); model unaffected"),
    ("paracrawl-bonus", "ParaCrawl-Bonus", "v9", "CC0 1.0"),
)


def clean(text: str) -> str:
    text = html.unescape(text)
    text = URL.sub(" ", text)
    text = TAGS.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def load_pairs(raw: Path, key: str) -> list[tuple[str, str]]:
    """Read one OPUS moses package as (vi, en) pairs.

    OPUS names the files ``<Corpus>.en-vi.en`` and ``<Corpus>.en-vi.vi``: the
    first suffix is the file's language, so the ``vi`` file pairs with the
    ``en`` file line by line.
    """
    dirs = [d for d in raw.iterdir() if d.is_dir() and d.name.startswith(key)]
    if not dirs:
        raise SystemExit(f"missing corpus {key} under {raw}; fetch it first")
    d = dirs[0]
    vi_files = sorted(d.glob("*.vi"))
    en_files = sorted(d.glob("*.en"))
    if not vi_files or not en_files:
        raise SystemExit(f"{d} does not hold a .en/.vi pair")
    vi_lines = vi_files[0].read_text(encoding="utf-8", errors="replace").splitlines()
    en_lines = en_files[0].read_text(encoding="utf-8", errors="replace").splitlines()
    if len(vi_lines) != len(en_lines):
        raise SystemExit(f"{d}: {len(vi_lines)} vi lines vs {len(en_lines)} en lines")
    return list(zip(vi_lines, en_lines, strict=True))


def acceptable(vi: str, en: str) -> bool:
    vw, ew = vi.split(), en.split()
    if not (MIN_WORDS <= len(vw) <= MAX_WORDS and MIN_WORDS <= len(ew) <= MAX_WORDS):
        return False
    ratio = len(ew) / len(vw)
    if not 0.4 <= ratio <= 2.5:
        return False
    if not VI_DIACRITIC.search(vi):
        return False
    # Alignment junk: a line that is mostly numbers or a lone word repeated.
    return not (len(vw) > 3 and len({w.lower() for w in vw}) == 1)


def register_ok(vi: str, en: str) -> bool:
    """Match the supplied corpus's measured shape, not just its vocabulary."""
    vw, ew = vi.split(), en.split()
    if not (
        REG_MIN_WORDS <= len(vw) <= REG_MAX_VI
        and REG_MIN_WORDS <= len(ew) <= REG_MAX_EN
    ):
        return False
    joined = f"{vi} {en}"
    if any(mark in joined for mark in BOILERPLATE):
        return False
    if any(mark in en for mark in NOT_ENGLISH):
        return False
    if ENUM.match(vi) or ENUM.match(en) or LABEL.match(vi) or LABEL.match(en):
        return False
    if TOO_NUMERIC.search(joined):
        return False
    if any(ch in joined for ch in "();"):
        return False
    tokens = {w.strip(".,!?'\"").lower() for w in ew}
    return bool(tokens & EN_FUNCTION)


def strip_label(text: str) -> str:
    return LABEL.sub("", text).strip()


def strip_source_punct(text: str) -> str:
    """Drop punctuation from the source side only.

    ASR emits unpunctuated text, so punctuated training source teaches the model
    to expect cues that never arrive. Target punctuation stays: it is what the
    clause cutter keys on and what the ship gate measures.
    """
    return SPACES.sub(" ", re.sub(r"[.,;:!?…]", " ", text)).strip()


def register_score(vi: str, en: str) -> int:
    low_vi, low_en = vi.lower(), en.lower()
    score = sum(1 for c in VI_CUES if c in low_vi) + sum(
        1 for c in EN_CUES if c in low_en
    )
    if en.rstrip().endswith("?"):
        score += 2  # 35/100 of the supplied gold pairs are questions
    if len(en.split()) <= 12:
        score += 1  # gold median is 8 English words
    return score


def dedupe(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out = []
    for vi, en in pairs:
        key = hashlib.blake2b(f"{vi}\u0000{en}".encode(), digest_size=16).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append((vi, en))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="eval_data/train_raw")
    ap.add_argument("--out", default="eval_data/train")
    ap.add_argument("--general", type=int, default=60000, help="general train pairs")
    ap.add_argument("--register", type=int, default=30000, help="register train pairs")
    ap.add_argument("--dev", type=int, default=2000)
    ap.add_argument("--min-register-score", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--comma-target",
        type=float,
        default=0.40,
        help="fraction of train_mix.tsv whose English side carries a comma (gold: 0.42)",
    )
    ap.add_argument(
        "--question-target",
        type=float,
        default=0.32,
        help="fraction of train_mix.tsv that is a question (gold: 0.35)",
    )
    ap.add_argument(
        "--require-target-terminal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop pairs whose English side has no . ? ! (gold is 100% punctuated)",
    )
    ap.add_argument(
        "--max-target-source-ratio",
        type=float,
        default=1.5,
        help="drop pairs whose English side is longer than this multiple of the source",
    )
    ap.add_argument(
        "--max-target-words",
        type=int,
        default=14,
        help="drop pairs whose English side exceeds this many words (gold mean 8.46, p95 15)",
    )
    ap.add_argument(
        "--keep-source-punct",
        type=float,
        default=0.4,
        help="fraction of rows that keep source punctuation; punctuation/case dropout",
    )
    ap.add_argument(
        "--lowercase-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="lowercase the source side to match ASR output",
    )
    args = ap.parse_args()

    raw, out = Path(args.raw_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    pool: list[tuple[str, str, str]] = []  # (vi, en, source)
    counts: dict[str, dict[str, int | str]] = {}
    dropped: dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for key, name, version, licence in SOURCES:
        pairs = load_pairs(raw, key)
        kept = 0
        for vi, en in pairs:
            vi, en = clean(vi), clean(en)
            vi, en = strip_label(vi), strip_label(en)
            if not vi or not en or not acceptable(vi, en):
                drop("length_or_junk")
                continue
            # The punctuation contract, applied to every output so train, dev
            # and inference share one convention.
            if args.require_target_terminal and not en.rstrip().endswith(TERMINAL):
                drop("target_unpunctuated")
                continue
            if len(en.split()) > args.max_target_source_ratio * len(vi.split()):
                drop("target_too_long")
                continue
            # Brevity is a training target, not a gate applied after the fact: the
            # first corrected lane emitted 16.66 words against a gold 8.46, which
            # is what a corpus averaging 10.59 teaches. The cap is the measured
            # lever for the single largest remaining gap.
            if len(en.split()) > args.max_target_words:
                drop("target_over_target_words")
                continue
            if args.lowercase_source:
                vi = vi.lower()
            if rng.random() >= args.keep_source_punct:
                vi = strip_source_punct(vi)
            if not vi:
                drop("source_empty_after_strip")
                continue
            pool.append((vi, en, name))
            kept += 1
        counts[name] = {
            "read": len(pairs),
            "kept": kept,
            "version": version,
            "licence": licence,
        }
        print(f"{name:16s} read {len(pairs):>9,}  kept {kept:>9,}  ({licence})")

    print(f"\npool after length/ratio/diacritic filters: {len(pool):,}")
    print("dropped by the punctuation contract:")
    for reason, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:26s} {n:>10,}")
    pool = [(vi, en, s) for vi, en, s in pool]
    uniq = dedupe([(vi, en) for vi, en, _ in pool])
    print(f"after dedupe: {len(uniq):,}")

    scored = [
        (register_score(vi, en), vi, en) for vi, en in uniq if register_ok(vi, en)
    ]
    register = [
        (vi, en) for score, vi, en in scored if score >= args.min_register_score
    ]
    rng.shuffle(register)
    register = register[: args.register]
    reg_set = set(register)
    general = [(vi, en) for vi, en in uniq if (vi, en) not in reg_set]
    rng.shuffle(general)
    general = general[: args.general + args.dev]
    dev, general = general[: args.dev], general[args.dev :]

    def write(name: str, rows: list[tuple[str, str]]) -> None:
        path = out / name
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for vi, en in rows:
                fh.write(f"{vi}\t{en}\n")
        print(f"wrote {path}  {len(rows):,} pairs")

    write("train_general.tsv", general)
    write("train_register.tsv", register)
    write("dev.tsv", dev)

    # Punctuation density is a training target, not a side effect. The register
    # slice carries commas at ~16% against the supplied gold's 42%, and the
    # pinned model already emits 45/100 - so training on the raw slice would
    # erode punctuation the model gets right today, and clause cutting depends
    # on it. The comma-bearing share is drawn from the FULL filtered pool: the
    # already-sampled general+register slice only holds ~22k comma rows, which
    # caps the achievable rate near 25% and silently misses the target.
    dev_set = set(dev)
    reg_seen = set(register)
    wide = [p for p in uniq if p not in dev_set and p not in reg_seen]

    def is_q(p: tuple[str, str]) -> bool:
        return p[1].rstrip().endswith("?")

    def is_c(p: tuple[str, str]) -> bool:
        return "," in p[1]

    def draw(n: int, comma_frac: float, question_frac: float) -> list[tuple[str, str]]:
        """Fill a cell-priority draw so BOTH marginals land near target.

        Drawing for one axis alone trades against the other: oversampling commas
        alone drops questions from 40% to 16% against a gold 35%, which trains
        the model to under-form questions. Filling comma-and-question pairs first
        satisfies both without duplicating rows.
        """
        want_c, want_q = int(n * comma_frac), int(n * question_frac)
        cells = {
            "both": [p for p in wide if is_c(p) and is_q(p)],
            "comma": [p for p in wide if is_c(p) and not is_q(p)],
            "question": [p for p in wide if not is_c(p) and is_q(p)],
            "plain": [p for p in wide if not is_c(p) and not is_q(p)],
        }
        for bucket in cells.values():
            rng.shuffle(bucket)
        picked: list[tuple[str, str]] = []
        have_c = have_q = 0

        def take(cell: str, k: int, *, adds_comma: bool, adds_q: bool) -> int:
            nonlocal have_c, have_q
            k = min(k, len(cells[cell]), n - len(picked))
            if k <= 0:
                return 0
            chunk = cells[cell][:k]
            del cells[cell][:k]
            picked.extend(chunk)
            have_c += k if adds_comma else 0
            have_q += k if adds_q else 0
            return k

        # Both first, then whichever marginal is further behind, then plain.
        while len(picked) < n:
            short_c = max(0, want_c - have_c)
            short_q = max(0, want_q - have_q)
            took = 0
            if short_c and short_q:
                took = take("both", min(short_c, short_q), adds_comma=True, adds_q=True)
            if not took and short_c:
                took = take("comma", short_c, adds_comma=True, adds_q=False)
                if not took:
                    # The pool cannot supply that many comma rows at this target
                    # length; stop asking rather than starve the whole slice.
                    have_c = want_c
            if not took and short_q:
                took = take("question", short_q, adds_comma=False, adds_q=True)
                if not took:
                    have_q = want_q
            if not took and not take(
                "plain", n - len(picked), adds_comma=False, adds_q=False
            ):
                break  # every cell is empty: the slice cannot reach n
        rng.shuffle(picked)
        return picked

    reg_comma = sum(1 for p in register if is_c(p))
    reg_q = sum(1 for p in register if is_q(p))
    n_all = len(general) + len(register)
    need_c = max(0, (int(n_all * args.comma_target) - reg_comma) / max(1, len(general)))
    need_q = max(0, (int(n_all * args.question_target) - reg_q) / max(1, len(general)))
    general = draw(len(general), min(1.0, need_c), min(1.0, need_q))
    if not general:
        raise SystemExit("the general slice drew no pairs; relax the contract filters")
    mixture = register + general
    rng.shuffle(mixture)
    write("train_mix.tsv", mixture)
    mix_comma = sum(1 for p in mixture if is_c(p)) / max(1, len(mixture))
    mix_q = sum(1 for p in mixture if is_q(p)) / max(1, len(mixture))
    if (
        abs(mix_comma - args.comma_target) > 0.05
        or abs(mix_q - args.question_target) > 0.10
    ):
        print(
            f"note: mixture landed at comma {mix_comma:.1%} / question {mix_q:.1%} "
            f"against targets {args.comma_target:.0%} / {args.question_target:.0%}; "
            "the filtered pool could not supply the oversampling",
        )

    def stats(rows: list[tuple[str, str]]) -> dict[str, float | int]:
        n = max(1, len(rows))
        words = [len(en.split()) for _, en in rows]
        return {
            "n": len(rows),
            "mean_en_words": round(sum(words) / n, 2),
            "terminal_rate": round(
                sum(1 for _, en in rows if en.rstrip().endswith(TERMINAL)) / n, 3
            ),
            "comma_rate": round(sum(1 for _, en in rows if "," in en) / n, 3),
            "question_rate": round(
                sum(1 for _, en in rows if en.rstrip().endswith("?")) / n, 3
            ),
        }

    profile = {
        "train_mix": stats(mixture),
        "train_register": stats(register),
        "train_general": stats(general),
        "dev": stats(dev),
    }
    print(
        "\nEnglish-side profile (gold: 100% terminal, 42% comma, 35% question, 8.46 words):"
    )
    for label, row in profile.items():
        print(
            f"  {label:16s} n={row['n']:>7,}  mean {row['mean_en_words']:5.2f}  "
            f"terminal {row['terminal_rate']:6.1%}  comma {row['comma_rate']:6.1%}  "
            f"question {row['question_rate']:6.1%}"
        )

    manifest = {
        "sources": counts,
        "filters": {
            "min_words": MIN_WORDS,
            "max_words": MAX_WORDS,
            "length_ratio": [0.4, 2.5],
            "register_lengths": {"vi_max": REG_MAX_VI, "en_max": REG_MAX_EN},
            "require_vi_diacritic": True,
            "min_register_score": args.min_register_score,
        },
        "outputs": {
            "train_general": len(general),
            "train_register": len(register),
            "train_mix": len(mixture),
            "dev": len(dev),
            "train_mix_comma_rate": round(mix_comma, 3),
            "train_mix_question_rate": round(mix_q, 3),
        },
        "punctuation_contract": {
            "require_target_terminal": args.require_target_terminal,
            "max_target_source_ratio": args.max_target_source_ratio,
            "max_target_words": args.max_target_words,
            "lowercase_source": args.lowercase_source,
            "keep_source_punct": args.keep_source_punct,
            "dropped": dropped,
        },
        "profile": profile,
        "not_included": {
            "tier_b_wikimatrix": "CC BY-SA 4.0 - waits on the share-alike ruling",
            "tier_c_authored": "written by hand, not machine-drafted",
            "tier_d_factory": "written by hand for the scripted scenario",
        },
        "seed": args.seed,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
