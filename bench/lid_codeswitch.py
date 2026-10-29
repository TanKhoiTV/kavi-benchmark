"""Does dual-confidence language detection survive code-switching?

The v2 pipeline chooses which way to translate from the two recognisers'
per-token confidence (ADR-033), and the risk named when that was decided is
code-switching: a Vietnamese speaker naming English products, or mixing both
languages inside one VAD segment. This measures the *mechanism's* exposure to
that, which is a different question from how often it happens.

Method. Real audio from the organizers' calibration set, both languages, spliced
inside a single segment: VI then EN, and EN then VI, at three duration-dominance
ratios. Each clip is decoded by both shipping checkpoints and each decode's mean
per-token log-probability is compared.

What the numbers can and cannot say. Splicing puts a hard boundary where real
code-switching has none, so it overstates how *cleanly* a switch can be heard,
and it says nothing about how often speech is mixed in the contest room. What it
does say is whether the margin degrades (safe: the ladder defers) or stays
confident while pointing at the wrong language (unsafe: the pipeline commits).

The checkpoints are staged on the device, not fetched here:

    adb pull /sdcard/Android/data/com.kavi.app/files/asr-vi-apache eval_data/asr/
    adb pull /sdcard/Android/data/com.kavi.app/files/asr-parakeet  eval_data/asr/

Usage:

    uv run --with sherpa-onnx==2.0.1 --with sherpa-onnx-core==2.0.1 \
        python -m bench.lid_codeswitch --pairs 40
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
VI_MODEL = _ROOT / "eval_data/asr/asr-vi-apache"
EN_MODEL = _ROOT / "eval_data/asr/asr-parakeet"
VI_WAVS = _ROOT / "eval_data/organizer/vi-en/wav"
EN_WAVS = _ROOT / "eval_data/organizer/en-vi/wav"

#: |margin| below this is "undetermined" - the ladder defers instead of committing.
#: 0.05 log-prob per token is roughly a fifth of the measured clean-audio margin,
#: so it flags clips whose evidence is weak without needing a calibrated threshold.
THIN = 0.05


@dataclass(frozen=True)
class Margins:
    """One clip, decoded by both recognisers."""

    vi_mean: float | None
    en_mean: float | None
    vi_tokens: int
    en_tokens: int

    @property
    def margin(self) -> float | None:
        """Positive favours Vietnamese, negative favours English; None if no evidence."""
        if self.vi_mean is None or self.en_mean is None:
            return None
        return self.vi_mean - self.en_mean

    @property
    def winner(self) -> str:
        """A model that emitted nothing has no evidence, so the other one wins."""
        if self.vi_mean is None and self.en_mean is None:
            return "undecided"
        if self.vi_mean is None:
            return "en"
        if self.en_mean is None:
            return "vi"
        return "vi" if self.margin is not None and self.margin >= 0 else "en"

    @property
    def thin(self) -> bool:
        margin = self.margin
        # Undecided is the thinnest possible margin: the ladder must defer.
        return margin is None or abs(margin) < THIN


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    samples, sr = sf.read(str(path), dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]
    return samples, sr


def load_recognizers():
    import sherpa_onnx

    def build(model_dir: Path, model_type: str | None = None):
        encoder = str(model_dir / "encoder.int8.onnx")
        # The VI export ships an f32 decoder, the parakeet export an int8 one.
        decoder = model_dir / "decoder.onnx"
        decoder_path = str(
            decoder if decoder.exists() else model_dir / "decoder.int8.onnx"
        )
        joiner = str(model_dir / "joiner.int8.onnx")
        tokens = str(model_dir / "tokens.txt")
        if model_type is None:
            return sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder_path,
                joiner=joiner,
                tokens=tokens,
                num_threads=2,
            )
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder_path,
            joiner=joiner,
            tokens=tokens,
            num_threads=2,
            model_type=model_type,
        )

    if not VI_MODEL.exists() or not EN_MODEL.exists():
        raise FileNotFoundError(
            f"ASR checkpoints not staged under {VI_MODEL.parent}/ - see this module's "
            "docstring for the two adb pulls. Refusing to guess at another copy: the "
            "point is to measure the checkpoints that ship."
        )
    return build(VI_MODEL), build(EN_MODEL, "nemo_transducer")


def _decode(recognizer, samples: np.ndarray, sr: int) -> tuple[float | None, int]:
    """Mean per-token log-probability, or None when the model emitted nothing.

    An empty hypothesis is not a missing measurement - it is the strongest
    evidence in the comparison. On pure English audio the Vietnamese transducer
    emits no tokens at all, and treating that as NaN threw the signal away.
    """
    stream = recognizer.create_stream()
    stream.accept_waveform(sr, samples)
    recognizer.decode_stream(stream)
    probs = list(stream.result.ys_log_probs)
    if not probs:
        return None, 0
    return float(np.mean(probs)), len(probs)


def margins_for(vi_rec, en_rec, samples: np.ndarray, sr: int) -> Margins:
    vi_mean, vi_tokens = _decode(vi_rec, samples, sr)
    en_mean, en_tokens = _decode(en_rec, samples, sr)
    return Margins(vi_mean, en_mean, vi_tokens, en_tokens)


def splice_once(
    first: np.ndarray, second: np.ndarray, sr: int, *, ratio: float, gap_ms: int
) -> np.ndarray:
    """`ratio` of the total duration from `first`, the remainder from `second`.

    The total is bounded by whichever clip runs out first, so both halves keep
    their natural duration ratio instead of one being looped or padded.
    """
    gap = np.zeros(int(sr * gap_ms / 1000), dtype=np.float32)
    total = min(len(first) / ratio, len(second) / (1.0 - ratio))
    first_take = min(len(first), max(1, int(round(ratio * total))))
    second_take = min(len(second), max(1, int(round((1.0 - ratio) * total))))
    return np.concatenate([first[:first_take], gap, second[:second_take]])


def summarise(rows: list[dict], label: str, expected: str | None) -> dict:
    margins = [r["margin"] for r in rows if r["margin"] is not None]
    out = {
        "condition": label,
        "n": len(rows),
        "margin_p10": float(np.percentile(margins, 10)) if margins else None,
        "margin_p50": float(np.percentile(margins, 50)) if margins else None,
        "margin_p90": float(np.percentile(margins, 90)) if margins else None,
        "vi_win_share": float(np.mean([r["winner"] == "vi" for r in rows])),
        "thin_share": float(np.mean([r["thin"] for r in rows])),
        "vi_empty_share": float(np.mean([r["vi_mean"] is None for r in rows])),
        "en_empty_share": float(np.mean([r["en_mean"] is None for r in rows])),
        "vi_tokens_median": float(np.median([r["vi_tokens"] for r in rows])),
        "en_tokens_median": float(np.median([r["en_tokens"] for r in rows])),
    }
    if expected:
        out["agrees_with_expected"] = float(
            np.mean([r["winner"] == expected for r in rows])
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs", type=int, default=40, help="clip pairs per condition"
    )
    parser.add_argument("--gap-ms", type=int, default=80, help="silence at the splice")
    parser.add_argument("--out", default="bench-results/lid-codeswitch/report.json")
    args = parser.parse_args()

    vi_rec, en_rec = load_recognizers()
    vi_paths = sorted(glob.glob(str(VI_WAVS / "*.wav")))[: args.pairs]
    en_paths = sorted(glob.glob(str(EN_WAVS / "*.wav")))[: args.pairs]
    if not vi_paths or not en_paths:
        raise FileNotFoundError(
            "organizer wavs not staged under eval_data/organizer/ - the calibration "
            "corpus is gitignored and not redistributed"
        )

    conditions: list[dict] = []

    # Reference points: clean audio, where the expected answer is unambiguous.
    for label, paths, expected in (
        ("clean VI", vi_paths, "vi"),
        ("clean EN", en_paths, "en"),
    ):
        rows = []
        for path in paths:
            samples, sr = read_wav(Path(path))
            m = margins_for(vi_rec, en_rec, samples, sr)
            rows.append(
                {
                    "margin": m.margin,
                    "winner": m.winner,
                    "thin": m.thin,
                    "vi_mean": m.vi_mean,
                    "en_mean": m.en_mean,
                    "vi_tokens": m.vi_tokens,
                    "en_tokens": m.en_tokens,
                }
            )
        conditions.append(summarise(rows, label, expected))

    # Code-switched constructions: both orders, three dominance ratios. `ratio` is
    # the share of the *first* clip, so the expected winner flips with it.
    for ratio in (0.3, 0.5, 0.7):
        for order, first_lang in (("vi-first", "vi"), ("en-first", "en")):
            second_lang = "en" if first_lang == "vi" else "vi"
            expected = first_lang if ratio >= 0.5 else second_lang
            rows = []
            for vi_path, en_path in zip(vi_paths, en_paths, strict=False):
                vi_samples, sr = read_wav(Path(vi_path))
                en_samples, _ = read_wav(Path(en_path))
                first, second = (
                    (vi_samples, en_samples)
                    if order == "vi-first"
                    else (en_samples, vi_samples)
                )
                mixed = splice_once(first, second, sr, ratio=ratio, gap_ms=args.gap_ms)
                m = margins_for(vi_rec, en_rec, mixed, sr)
                rows.append(
                    {
                        "margin": m.margin,
                        "winner": m.winner,
                        "thin": m.thin,
                        "vi_mean": m.vi_mean,
                        "en_mean": m.en_mean,
                        "vi_tokens": m.vi_tokens,
                        "en_tokens": m.en_tokens,
                    }
                )
            conditions.append(
                summarise(
                    rows,
                    f"mixed {int(ratio * 100)}/{int((1 - ratio) * 100)} {order}",
                    expected,
                )
            )

    print(
        f"{'condition':26} {'n':>3} {'margin p10':>10} {'p50':>7} {'p90':>7} "
        f"{'VI wins':>8} {'thin':>6} {'VI null':>8} {'EN null':>8} {'agrees':>7}"
    )
    for c in conditions:
        agrees = c.get("agrees_with_expected")

        def fmt(value: float | None, width: int = 7) -> str:
            return f"{'-':>{width}}" if value is None else f"{value:>{width}.3f}"

        print(
            f"{c['condition']:26} {c['n']:>3} {fmt(c['margin_p10'], 10)} "
            f"{fmt(c['margin_p50'])} {fmt(c['margin_p90'])} "
            f"{c['vi_win_share']:>7.0%} {c['thin_share']:>6.0%} "
            f"{c['vi_empty_share']:>8.0%} {c['en_empty_share']:>8.0%} "
            f"{(agrees if agrees is not None else float('nan')):>7.0%}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "measured_on": str(date.today()),
                "provenance": {
                    "vi_checkpoint": str(VI_MODEL.relative_to(_ROOT)),
                    "en_checkpoint": str(EN_MODEL.relative_to(_ROOT)),
                    "signal": "mean per-token log-probability (ys_log_probs), sherpa-onnx 2.0.1",
                    "construction": "organizer wavs spliced inside one segment",
                    "gap_ms": args.gap_ms,
                    "pairs_per_condition": args.pairs,
                    "thin_threshold": THIN,
                    "synthetic": "spliced, not naturally code-switched speech",
                },
                "conditions": conditions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nartifact -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
