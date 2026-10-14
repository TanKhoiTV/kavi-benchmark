"""MT candidate: M2M-100 (facebook/m2m100_418M) vi->en via CTranslate2 int8 (offline).

Multilingual encoder-decoder. Unlike Opus-MT (fixed vi->en pair), M2M-100
requires target_prefix to force English output — omitting it produces a
random language regardless of beam_size.
"""

from __future__ import annotations

from pathlib import Path

from ..adapters import Candidate
from ..schema import EvalItem

_ROOT = Path(__file__).resolve().parents[2]
M2M_MODEL_DIR = _ROOT / "models" / "m2m100-418m-ct2-int8"


class M2M100MTCandidate(Candidate):
    stage = "MT"
    id = "m2m100-vi-en-ct2-cpu"

    def __init__(
        self, model_path: str | None = None, config: dict | None = None
    ) -> None:
        import ctranslate2
        from transformers import AutoTokenizer

        cfg = config or {}
        self._decode_kwargs = dict(
            beam_size=cfg.get("beam_size", 4),
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            max_decoding_length=256,
        )

        model_dir = Path(model_path) if model_path else M2M_MODEL_DIR
        if not model_dir.exists():
            raise FileNotFoundError(f"M2M-100 model not found at {model_dir}")
        self.translator = ctranslate2.Translator(
            str(model_dir), device="cpu", compute_type="int8"
        )
        # M2M-100 uses the original HF tokenizer (not 2 separate .spm files like Opus-MT)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/m2m100_418M", src_lang="vi"
        )
        # Pre-resolve the English language token for use in every _infer call
        self._en_token = self.tokenizer.lang_code_to_token["en"]

    def _infer(self, item: EvalItem) -> tuple[str | None, str | None]:
        text = item.resolve_input()
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        # CRITICAL: target_prefix forces English output. Without this,
        # M2M-100 may translate to a random language, producing BLEU ~0
        # regardless of beam_size.
        target_prefix = [[self._en_token]]
        results = self.translator.translate_batch(
            [tokens], target_prefix=target_prefix, **self._decode_kwargs
        )
        # Strip the leading language token from the output before decoding
        translation = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(results[0].hypotheses[0][1:])
        )
        return translation, None
