"""Off-device scorer: WER/CER (jiwer), BLEU (sacrebleu), RTF; COMET deferred to v1.

Scoring stays off the device (host-side). COMET and human MOS are Phase 7 (v1);
the v0 harness intentionally gates on RTF/turnaround/WER/BLEU/RAM only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import EvalItem, StageResult


@dataclass
class Metrics:
    wer: float | None = None
    cer: float | None = None
    bleu: float | None = None
    rtf: float | None = None
    notes: list[str] = field(default_factory=list)


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
