"""ASR candidate: Zipformer via sherpa-onnx (CPU).

Zipformer is a transducer-based ASR model that runs on sherpa-onnx,
not CTranslate2. Each language has its own checkpoint (single-language).

Beam concept differs from CT2/Transformer models:
- beam=1 -> greedy_search (true greedy decoding)
- beam>=2 -> modified_beam_search with max_active_paths=<beam>

This mapping is internal to the candidate. The config interface
(beam_size from config-override) stays consistent with Whisper/Moonshine.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..adapters import Candidate
from ..schema import EvalItem

_ROOT = Path(__file__).resolve().parents[2]


def decoding_method_for_beam(beam: int) -> str:
    """Map a beam width to a sherpa-onnx decoding method.

    beam=1 -> ``greedy_search`` (true greedy; mathematically different from
    modified_beam_search with max_active_paths=1). beam>=2 ->
    ``modified_beam_search`` with ``max_active_paths=beam``.
    """
    return "greedy_search" if beam == 1 else "modified_beam_search"


def _resolve_model_file(model_dir: Path, component: str) -> Path:
    """Pick one ONNX file for ``component`` deterministically.

    ``Path.glob`` yields filesystem order, so taking ``matches[0]`` lets a
    directory holding both a quantised and an fp32 build decide which model is
    benchmarked while the candidate id stays the same — a silent swap that makes
    a result irreproducible. The trained ``-epoch-N-avg-N`` exports ship exactly
    such pairs, so when more than one file matches, prefer the int8 build
    explicitly and refuse to guess at anything more ambiguous than that.
    """
    matches = sorted(model_dir.glob(f"{component}*.onnx"))
    if not matches:
        raise FileNotFoundError(
            f"no {component}*.onnx under {model_dir}; expected {component}.onnx "
            f"or {component}.int8.onnx"
        )
    if len(matches) == 1:
        return matches[0]
    int8 = [m for m in matches if ".int8.onnx" in m.name]
    if len(int8) == 1:
        return int8[0]
    raise ValueError(
        f"{model_dir} holds several {component}*.onnx "
        f"({[m.name for m in matches]}); point model_path at a directory with a "
        f"single intended build so the benchmark is reproducible"
    )


class ZipformerCandidateBase(Candidate):
    stage = "ASR"
    _model_dir_attr: str = ""
    _expected_lang: str = ""

    def __init__(
        self, model_path: str | None = None, config: dict | None = None
    ) -> None:
        import sherpa_onnx

        cfg = config or {}
        beam = cfg.get("beam_size", 4)

        decoding_method = decoding_method_for_beam(beam)

        model_dir = Path(model_path) if model_path else Path(self._model_dir_attr)
        # int8 encoder/joiner with an fp32 fallback, resolved deterministically
        # rather than by glob order.
        encoder = _resolve_model_file(model_dir, "encoder")
        decoder = _resolve_model_file(model_dir, "decoder")
        joiner = _resolve_model_file(model_dir, "joiner")

        tokens = model_dir / "tokens.txt"

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            decoding_method=decoding_method,
            max_active_paths=beam,  # ignored when decoding_method="greedy_search"
            provider="cpu",
        )

    def _infer(self, item: EvalItem) -> tuple[str | None, str | None]:
        lang = getattr(item, "language", None)
        if lang != self._expected_lang:
            raise ValueError(
                f"{type(self).__name__} received a non-{self._expected_lang} item: "
                f"{item.id} (language={lang!r})"
            )

        audio_path = item.audio_ref or item.input_text
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"audio file not found: {audio_path!r}")

        import soundfile as sf

        samples, sample_rate = sf.read(audio_path)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples.astype(np.float32))
        self.recognizer.decode_stream(stream)

        text = stream.result.text.strip()
        return text or "", None


class ZipformerViCandidate(ZipformerCandidateBase):
    id = "zipformer-vi-30m-sherpa-onnx-cpu"
    _model_dir_attr = str(_ROOT / "models" / "sherpa-onnx-zipformer-vi-30M-int8")
    _expected_lang = "vi"


class ZipformerEnCandidate(ZipformerCandidateBase):
    id = "zipformer-en-sherpa-onnx-cpu"
    _model_dir_attr = str(_ROOT / "models" / "sherpa-onnx-zipformer-en")
    _expected_lang = "en"
