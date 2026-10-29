"""Score MT on the organizers' register pairs, with BLEU, chrF++ and COMET.

The 200 organizer clips are the calibration set every claim in this project is
quoted against, but the harness could not read them: the register MT numbers in
`docs/reference/mt-en-vi-baseline.md` came from scripts that lived outside every
repository. This module puts that path inside the repo, so the number is
reproducible from a clone plus the gitignored `eval_data/organizer/` tree.

It translates the sheets' own source column, so it measures **MT only** - that is
the condition the published register BLEU figures were measured under, and why
the voice-to-voice numbers (which need the phone) are not reproducible here.

    uv run python -m bench.register_eval                      # both directions
    uv run python -m bench.register_eval --direction en-vi --beam 3
    uv run python -m bench.register_eval --no-comet           # skip the 2 GB model

The sheets are not redistributed (see `NOTICE`), so a missing sheet is a named
error rather than a skip: silently scoring zero items would look like a pass.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .decode import max_decoding_length
from .scorer import comet_scores

_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = _ROOT / "models" / "m2m100-418m-ct2-int8"
#: Upstream tokenizer for the language-token vocabulary; resolved from the local
#: Hugging Face cache when present, so scoring works without a network.
TOKENIZER = "facebook/m2m100_418M"


@dataclass(frozen=True)
class Register:
    """One direction of the organizers' calibration set."""

    direction: str
    sheet: Path
    src_lang: str
    tgt_lang: str
    #: Recorded in the artifact: a normalized source column is not the same input
    #: distribution as prose, and its BLEU is therefore not comparable across sets.
    source_note: str


REGISTERS: dict[str, Register] = {
    "vi-en": Register(
        direction="vi-en",
        sheet=_ROOT / "eval_data/organizer/vi-en/OneVoice_V_to_E_samples_100.xlsx",
        src_lang="vi",
        tgt_lang="en",
        source_note="sheet column 'Tiếng Việt': scripted travel register",
    ),
    "en-vi": Register(
        direction="en-vi",
        sheet=_ROOT / "eval_data/organizer/en-vi/OneVoice_E_to_V_samples_100.xlsx",
        src_lang="en",
        tgt_lang="vi",
        source_note=(
            "sheet column 'Tiếng Anh (normalized)': numbers spelled out by the "
            "organizers' normaliser, so not natural written English"
        ),
    ),
}


@dataclass(frozen=True)
class Pair:
    item_id: str
    source: str
    reference: str


def load_pairs(register: Register, limit: int | None = None) -> list[Pair]:
    """Read `id, source, reference` from the register sheet.

    Columns are resolved by header position rather than hard-coded indices, and the
    header is asserted: a re-issued sheet with a different shape must fail here
    rather than score the wrong column.
    """
    import openpyxl

    if not register.sheet.exists():
        raise FileNotFoundError(
            f"{register.direction}: missing {register.sheet}. The organizer corpus is "
            "not redistributed - stage it under eval_data/organizer/ to score this "
            "direction."
        )

    sheet = openpyxl.load_workbook(register.sheet, read_only=True).active
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        raise ValueError(f"{register.direction}: {register.sheet.name} is empty")

    header = [str(cell).strip() if cell is not None else "" for cell in rows[0][:3]]
    if header[0].upper() != "ID":
        raise ValueError(
            f"{register.direction}: expected an 'ID' first column, got {header[0]!r}"
        )

    pairs: list[Pair] = []
    for row in rows[1:]:
        if not row or row[0] is None:
            continue
        item_id, source, reference = (str(row[0]).strip(), row[1], row[2])
        if not source or not reference:
            continue
        pairs.append(Pair(item_id, str(source).strip(), str(reference).strip()))
        if limit is not None and len(pairs) >= limit:
            break
    if not pairs:
        raise ValueError(
            f"{register.direction}: no usable pairs in {register.sheet.name}"
        )
    return pairs


def translate(
    register: Register,
    pairs: list[Pair],
    *,
    beam_size: int = 1,
    model_dir: Path = MODEL_DIR,
) -> tuple[list[str], float]:
    """Translate the source column with the shipped M2M-100 export.

    Decode settings mirror `bench/candidates/m2m_mt.py` - the guard, the
    repetition penalties and the stripped target-prefix token - because those are
    the conditions the published register figures were produced under.
    """
    import ctranslate2
    from transformers import AutoTokenizer

    if not model_dir.exists():
        raise FileNotFoundError(
            f"M2M-100 not found at {model_dir}; convert facebook/m2m100_418M to "
            "CTranslate2 int8 first (see NOTICE 'Fetching')."
        )

    translator = ctranslate2.Translator(
        str(model_dir), device="cpu", compute_type="int8"
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, src_lang=register.src_lang)
    target_token = tokenizer.lang_code_to_token[register.tgt_lang]

    hypotheses: list[str] = []
    started = time.monotonic()
    for pair in pairs:
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(pair.source))
        result = translator.translate_batch(
            [tokens],
            target_prefix=[[target_token]],
            max_decoding_length=max_decoding_length(len(tokens)),
            beam_size=beam_size,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
        # The forced prefix arrives as a decode step; leaving it in would put a
        # language token in front of every hypothesis.
        hypotheses.append(
            tokenizer.decode(
                tokenizer.convert_tokens_to_ids(result[0].hypotheses[0][1:])
            ).strip()
        )
    return hypotheses, time.monotonic() - started


def score(
    register: Register,
    pairs: list[Pair],
    hypotheses: list[str],
    *,
    with_comet: bool,
) -> dict:
    """Corpus BLEU, chrF++ and optionally COMET, plus the per-item rows."""
    import sacrebleu

    references = [pair.reference for pair in pairs]
    sources = [pair.source for pair in pairs]

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

    comet_system: float | None = None
    comet_items: list[float | None] = [None] * len(hypotheses)
    if with_comet:
        scores = comet_scores(sources, hypotheses, references)
        comet_system = scores.system
        comet_items = list(scores.per_item)

    sentence_bleu = [
        sacrebleu.sentence_bleu(hyp, [ref]).score
        for hyp, ref in zip(hypotheses, references, strict=True)
    ]
    return {
        "corpus_bleu": bleu.score,
        "corpus_chrf_pp": chrf.score,
        "comet": comet_system,
        "items": [
            {
                "id": pair.item_id,
                "source": pair.source,
                "reference": pair.reference,
                "hypothesis": hyp,
                "sentence_bleu": sentence_bleu[i],
                "comet": comet_items[i],
            }
            for i, (pair, hyp) in enumerate(zip(pairs, hypotheses, strict=True))
        ],
    }


def run(
    direction: str,
    *,
    beam_size: int,
    limit: int | None,
    with_comet: bool,
    out_dir: Path,
) -> dict:
    register = REGISTERS[direction]
    pairs = load_pairs(register, limit=limit)
    hypotheses, elapsed = translate(register, pairs, beam_size=beam_size)
    metrics = score(register, pairs, hypotheses, with_comet=with_comet)

    artifact = {
        "direction": direction,
        "measured_on": str(date.today()),
        # INV-8: a number without its provenance is not a measurement.
        "provenance": {
            "model": f"facebook/m2m100_418M -> CTranslate2 int8 ({MODEL_DIR.name})",
            "tokenizer": TOKENIZER,
            "decode": {
                "beam_size": beam_size,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "max_decoding_length": "bench.decode.max_decoding_length (1.5x, 48-256)",
                "target_prefix": register.tgt_lang,
            },
            "input": register.source_note,
            "sheet": str(register.sheet.relative_to(_ROOT)),
            "n_items": len(pairs),
            "metrics": "sacrebleu 13a BLEU, chrF++ (word_order=2), COMET wmt22-comet-da",
            "mt_seconds": round(elapsed, 2),
            "hardware": "host CPU, int8",
        },
        **metrics,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{direction}.json"
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction",
        choices=[*REGISTERS, "both"],
        default="both",
        help="which register to score (default: both)",
    )
    parser.add_argument("--beam", type=int, default=1, help="beam width (shipped: 1)")
    parser.add_argument(
        "--limit", type=int, default=None, help="score the first N pairs"
    )
    parser.add_argument(
        "--no-comet",
        action="store_true",
        help="skip COMET (avoids a ~2 GB checkpoint download)",
    )
    parser.add_argument(
        "--out", default="bench-results/register-mt", help="artifact directory"
    )
    args = parser.parse_args()

    directions = list(REGISTERS) if args.direction == "both" else [args.direction]
    out_dir = Path(args.out)
    print(f"{'direction':10} {'n':>4} {'BLEU':>7} {'chrF++':>7} {'COMET':>7}")
    for direction in directions:
        artifact = run(
            direction,
            beam_size=args.beam,
            limit=args.limit,
            with_comet=not args.no_comet,
            out_dir=out_dir,
        )
        comet = artifact["comet"]
        print(
            f"{direction:10} {len(artifact['items']):>4} "
            f"{artifact['corpus_bleu']:>7.2f} {artifact['corpus_chrf_pp']:>7.2f} "
            f"{comet:>7.4f}"
            if comet is not None
            else f"{direction:10} {len(artifact['items']):>4} "
            f"{artifact['corpus_bleu']:>7.2f} {artifact['corpus_chrf_pp']:>7.2f} "
            f"{'-':>7}"
        )
    print(f"artifacts -> {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
