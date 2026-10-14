"""ASR candidate: Moonshine Tiny via HuggingFace Transformers (CPU).

Moonshine is a family of single-language speech-to-text models.
Unlike Whisper (1 multilingual model), each language has its own weights,
so we need separate candidate classes for vi and en.

Runtime: PyTorch Transformers (no CTranslate2 support — PR #1808 open since
Oct 2024). Weights are fetched from HuggingFace at setup (not vendored).
"""

from __future__ import annotations

import os

import torch

from ..adapters import Candidate
from ..schema import EvalItem


class MoonshineTinyViCandidate(Candidate):
    stage = "ASR"
    id = "moonshine-tiny-vi-hf-cpu"

    def __init__(
        self, model_path: str | None = None, config: dict | None = None
    ) -> None:
        from transformers import (
            AutoProcessor,
            MoonshineForConditionalGeneration,
        )

        cfg = config or {}
        beam = cfg.get("beam_size", 1)

        # Map beam_size from config-override to num_beams in generate().
        # Moonshine uses standard HF generate(), so num_beams is the correct param.
        self._decode_kwargs = dict(
            num_beams=beam,
            early_stopping=True,
            no_repeat_ngram_size=3,
            max_length=448,  # increased from default 194 to handle 15s FLEURS utterances
            use_cache=True,
        )

        model_name = model_path or "usefulsensors/moonshine-tiny-vi"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MoonshineForConditionalGeneration.from_pretrained(model_name)

    def _infer(self, item: EvalItem) -> tuple[str | None, str | None]:
        lang = getattr(item, "language", None)
        if lang != "vi":
            raise ValueError(
                f"MoonshineTinyViCandidate received a non-vi item: "
                f"{item.id} (language={lang!r})"
            )

        audio_path = item.audio_ref or item.input_text
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"audio file not found: {audio_path!r}")

        # Load audio as mono 16kHz
        import soundfile as sf

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo -> mono
        # The processor is told 16 kHz below, so the audio must actually be at
        # 16 kHz. Every manifest shipped today is, but a VSS or external
        # manifest need not be, and mislabelling the rate yields a transcript
        # that is wrong rather than an error.
        if sr != 16000:
            import torchaudio

            audio = torchaudio.functional.resample(
                torch.from_numpy(audio).float(), sr, 16000
            ).numpy()

        # Process and generate
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_values,
                attention_mask=inputs.attention_mask,
                **self._decode_kwargs,
            )

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip(), None


class MoonshineTinyEnCandidate(Candidate):
    stage = "ASR"
    id = "moonshine-tiny-en-hf-cpu"

    def __init__(
        self, model_path: str | None = None, config: dict | None = None
    ) -> None:
        from transformers import (
            AutoProcessor,
            MoonshineForConditionalGeneration,
        )

        cfg = config or {}
        beam = cfg.get("beam_size", 1)

        self._decode_kwargs = dict(
            num_beams=beam,
            early_stopping=True,
            no_repeat_ngram_size=3,
            max_length=448,
            use_cache=True,
        )

        model_name = model_path or "usefulsensors/moonshine-tiny"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MoonshineForConditionalGeneration.from_pretrained(model_name)

    def _infer(self, item: EvalItem) -> tuple[str | None, str | None]:
        lang = getattr(item, "language", None)
        if lang != "en":
            raise ValueError(
                f"MoonshineTinyEnCandidate received a non-en item: "
                f"{item.id} (language={lang!r})"
            )

        audio_path = item.audio_ref or item.input_text
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"audio file not found: {audio_path!r}")

        import soundfile as sf

        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # See the note in MoonshineTinyViCandidate._infer: the declared rate
        # must match the audio, not merely be asserted to the processor.
        if sr != 16000:
            import torchaudio

            audio = torchaudio.functional.resample(
                torch.from_numpy(audio).float(), sr, 16000
            ).numpy()

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_values,
                attention_mask=inputs.attention_mask,
                **self._decode_kwargs,
            )

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip(), None
