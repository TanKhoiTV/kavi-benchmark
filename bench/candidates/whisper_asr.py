"""ASR candidate: Whisper Small via faster-whisper (CPU int8).

v0 CPU candidate for the ASR stage. Covers vi+en. Needs the Whisper Small
weights fetched once from Hugging Face (setup, not runtime); the on-device
QNN-quantized Whisper-Small is a later (Phase 4) candidate.
"""

from __future__ import annotations

import os

from ..adapters import Candidate
from ..schema import EvalItem


class WhisperASRCandidate(Candidate):
    stage = "ASR"
    id = "whisper-small-faster-whisper-cpu"

    def __init__(
        self, model_path: str | None = None, config: dict | None = None
    ) -> None:
        from faster_whisper import WhisperModel

        cfg = config or {}
        # beam_size is the swept variable, fixed for the whole run. Language is
        # NOT set here — it is a per-item attribute (vi or en audio), not a
        # fixed configuration variable.
        self._decode_kwargs = dict(
            beam_size=cfg.get("beam_size", 5),
            temperature=0,
            condition_on_previous_text=False,
            without_timestamps=True,
        )

        # model_path here is a faster-whisper model size/name (e.g. "small").
        self.model = WhisperModel(
            model_size_or_path=cfg.get("model_size", "small"),
            device=cfg.get("device", "cpu"),
            compute_type=cfg.get("compute_type", "int8"),
        )

    def _infer(self, item: EvalItem) -> tuple[str | None, str | None]:
        audio_path = item.audio_ref or item.input_text
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"audio file not found: {audio_path!r}")

        # Language comes from the item — forced, not auto-detected.
        lang = getattr(item, "language", None)
        if lang not in ("vi", "en"):
            raise ValueError(
                f"missing/invalid language tag on item {item.id}: {lang!r}"
            )

        segments, _info = self.model.transcribe(
            audio_path, language=lang, **self._decode_kwargs
        )
        text = "".join(seg.text for seg in segments).strip()
        return text or "", None
