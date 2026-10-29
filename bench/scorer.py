"""Off-device scorer: WER/CER (jiwer), BLEU (sacrebleu), COMET (unbabel-comet), RTF.

Scoring stays off the device (host-side). COMET and human MOS were deferred to v1
by the v0 harness, and COMET is now implemented here because the contest's Luma
criteria name BLEU **and** COMET while only BLEU had ever been reported. It stays
optional and lazy: the checkpoint is ~2 GB, so a run that does not ask for it
never pays for it, and `score_item` keeps its v0 behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import EvalItem, StageResult


@dataclass
class Metrics:
    wer: float | None = None
    cer: float | None = None
    bleu: float | None = None
    rtf: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class CometScore:
    """COMET on its own 0-1 scale: corpus score plus one score per item."""

    system: float
    per_item: list[float]


#: COMET checkpoints are ~2 GB and a load costs tens of seconds, so one model is
#: kept per process rather than per call. Keyed by name to keep the choice visible
#: in the artifact that reports the score. Typed `Any` because `comet` ships no type
#: stubs; a Protocol here would only restate a signature the package owns.
_COMET_CACHE: dict[str, Any] = {}


def comet_scores(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    model: str = "Unbabel/wmt22-comet-da",
) -> CometScore:
    """Corpus and per-item COMET for one direction.

    `wmt22-comet-da` is the reference-based default: it is the checkpoint the
    contest's "BLEU + COMET" phrasing implies, and it needs the source text, which
    the register sheets supply. Scores are on COMET's own 0-1 scale and are not
    comparable to BLEU.
    """
    if not (len(sources) == len(hypotheses) == len(references)):
        raise ValueError("sources, hypotheses and references must be the same length")

    if model not in _COMET_CACHE:
        from comet import download_model, load_from_checkpoint

        _COMET_CACHE[model] = load_from_checkpoint(download_model(model))
    scorer = _COMET_CACHE[model]

    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references, strict=True)
    ]
    # batch_size=8 is a CPU-memory compromise; gpus=0 keeps the run deterministic
    # and offline after the one-time checkpoint download.
    output = scorer.predict(data, batch_size=8, gpus=0)
    return CometScore(
        system=float(output.system_score), per_item=[float(s) for s in output.scores]
    )


def score_item(
    item: EvalItem, result: StageResult, audio_duration_s: float | None = None
) -> Metrics:
    metrics = Metrics()
    if result.error:
        metrics.notes.append(f"error: {result.error}")
        return metrics

    if item.stage == "ASR":
        ref = item.resolve_reference()
        if not ref:
            metrics.notes.append("no reference text -> skipped WER/CER")
            return metrics
        hyp = result.output_text or ""
        try:
            import jiwer

            # Normalize case: Zipformer outputs ALL CAPS, references are
            # lowercase. jiwer is case-sensitive, so "THEY" != "they" gives
            # misleading 100% WER. Lowercase both sides for fair scoring across
            # all ASR candidates.
            ref_norm = ref.lower()
            hyp_norm = hyp.lower()
            metrics.wer = jiwer.wer(ref_norm, hyp_norm)
            metrics.cer = jiwer.cer(ref_norm, hyp_norm)
        except Exception as exc:  # noqa: BLE001
            metrics.notes.append(f"wer/cer failed: {exc}")
    elif item.stage == "MT":
        ref = item.resolve_reference()
        if not ref:
            metrics.notes.append("no reference text -> skipped BLEU")
            return metrics
        hyp = result.output_text or ""
        try:
            import sacrebleu

            metrics.bleu = sacrebleu.corpus_bleu([hyp], [[ref]]).score
        except Exception as exc:  # noqa: BLE001
            metrics.notes.append(f"bleu failed: {exc}")
    elif item.stage == "TTS":
        metrics.notes.append("TTS MOS deferred to v1 (DNSMOS / human panel)")

    if audio_duration_s and result.latency_s:
        metrics.rtf = result.latency_s / audio_duration_s
    return metrics
