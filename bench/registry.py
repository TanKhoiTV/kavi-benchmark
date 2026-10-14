"""Candidate registry: candidate_id -> (class, default model_path, default config)."""

from __future__ import annotations

from .candidates.hy_mt_hf import HyMT15MTCandidate
from .candidates.m2m_mt import M2M100MTCandidate
from .candidates.moonshine_asr import (
    MoonshineTinyEnCandidate,
    MoonshineTinyViCandidate,
)
from .candidates.opusmt_mt import OpusMTMTCandidate
from .candidates.piper_tts import PiperTTSCandidate
from .candidates.qnn_opusmt_mt import QnnOpusMTMTCandidate

# qnn_whisper_asr / qnn_piper_tts: superseded by ADR-008/009 — kept registered
# as no-op stubs for the record; only QnnOpusMTMTCandidate is on the v1 path.
from .candidates.qnn_piper_tts import QnnPiperTTSCandidate
from .candidates.qnn_whisper_asr import QnnWhisperASRCandidate
from .candidates.rtranslator import RTranslatorCandidate
from .candidates.whisper_asr import WhisperASRCandidate
from .candidates.zipformer_asr import ZipformerEnCandidate, ZipformerViCandidate
from .schema import EvalItem

RegistryEntry = tuple[type, str | None, dict]

# Default beam_size per candidate = optimal value from the exploratory beam
# sweep phase (see docs/benchmark/register-missing-candidates.md §8 D3, and
# bench-results/archive/beam-sweep-* / bench-results/fluers/).
# Dict order matters: default_candidate_id_for_stage() returns the FIRST
# candidate whose stage matches, so keep Whisper first (ASR default) and
# Opus-MT before M2M (MT default) — D2.
REGISTRY: dict[str, RegistryEntry] = {
    WhisperASRCandidate.id: (WhisperASRCandidate, None, {"beam_size": 2}),
    MoonshineTinyViCandidate.id: (MoonshineTinyViCandidate, None, {"beam_size": 1}),
    MoonshineTinyEnCandidate.id: (MoonshineTinyEnCandidate, None, {"beam_size": 4}),
    ZipformerViCandidate.id: (ZipformerViCandidate, None, {"beam_size": 5}),
    ZipformerEnCandidate.id: (ZipformerEnCandidate, None, {"beam_size": 4}),
    OpusMTMTCandidate.id: (OpusMTMTCandidate, None, {"beam_size": 5}),
    M2M100MTCandidate.id: (M2M100MTCandidate, None, {"beam_size": 4}),
    HyMT15MTCandidate.id: (HyMT15MTCandidate, None, {"beam_size": 5}),
    PiperTTSCandidate.id: (PiperTTSCandidate, None, {}),
    QnnWhisperASRCandidate.id: (QnnWhisperASRCandidate, None, {}),
    QnnOpusMTMTCandidate.id: (QnnOpusMTMTCandidate, None, {}),
    QnnPiperTTSCandidate.id: (QnnPiperTTSCandidate, None, {}),
    RTranslatorCandidate.id: (RTranslatorCandidate, None, {}),
}


def default_candidate_id_for_stage(stage: str) -> str | None:
    """Return the registered candidate id for `stage` (first match), or None.

    Note: with multiple candidates per stage, this returns the FIRST registered
    match (dict insertion order). Explicit `candidate_id` in the manifest or
    `--candidate` on the CLI is the deterministic way to pick a specific model.
    """
    for cid, (cls, _, _) in REGISTRY.items():
        if getattr(cls, "stage", None) == stage:
            return cid
    return None


def build_candidate(item: EvalItem):
    cid = item.candidate_id or default_candidate_id_for_stage(item.stage)
    if cid is None or cid not in REGISTRY:
        raise KeyError(
            f"no candidate for item {item.id!r} (stage={item.stage!r}, "
            f"candidate_id={item.candidate_id!r}); known: {sorted(REGISTRY)}"
        )
    cls, default_model, default_cfg = REGISTRY[cid]
    model_path = item.model_path if item.model_path is not None else default_model
    config = {**default_cfg, **(item.config or {})}
    return cls(model_path=model_path, config=config)
