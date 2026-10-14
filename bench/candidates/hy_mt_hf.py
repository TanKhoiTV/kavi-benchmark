"""MT candidate: Tencent HY-MT1.5-1.8B vi->en via HF Transformers (CPU, bf16).

Issue #91 blocked the CTranslate2 conversion of `tencent/HY-MT1.5-1.8B`
(custom HunYuanDenseV1 arch, `dynamic` RoPE scaling, QK-norm — all
unsupported by CT2's LLaMA converter; verified against CT2 4.8.0). This
candidate runs the model with native HF `generate` instead.

Benchmark implications (from the issue's own measurements):
  - BLEU is fully valid and comparable with the CT2 candidates.
  - latency is NOT comparable (HF eager CPU ~19 s/inference vs CT2 int8
    ~1 s) — always flag Hy-MT latency separately and never average it
    into CT2-based latency comparisons.
  - peak RAM ~3.5-4 GB (bf16), well above the on-device budget.

Decoding is deterministic beam search (do_sample=False) so BLEU stays
comparable with Opus-MT / M2M-100, both of which run CT2 beam search.
The prompt follows the model card's XX->XX template (no ZH involved):
    "Translate the following segment into English, without additional
     explanation.\\n\\n{source_text}"
and the chat template is applied exactly as in the README snippet
(add_generation_prompt=False).
"""

from __future__ import annotations

from ..adapters import Candidate
from ..schema import EvalItem

# HF repo id; weights resolve from the local HF cache
# (~/.cache/huggingface/hub/models--tencent--HY-MT1.5-1.8B).
HYMT_HF_ID = "tencent/HY-MT1.5-1.8B"

# token ids from config.json (also in generation_config.json)
_BOS = 120000
_EOS = 120020
_PAD = 120002


class HyMT15MTCandidate(Candidate):
    stage = "MT"
    id = "hy-mt1.5-1.8b-hf-cpu"

    def __init__(
        self, model_path: str | None = None, config: dict | None = None
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = config or {}
        # Beam search mirrors the CT2 candidates (Opus beam 5 / M2M beam 4);
        # repetition controls match the m2m/opus decode kwargs. An explicit
        # GenerationConfig replaces the model's sampling config (top_k/top_p,
        # do_sample=True) so decoding is deterministic beam search.
        from transformers import GenerationConfig

        self._gen_kwargs = dict(
            num_beams=cfg.get("beam_size", 5),
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            max_new_tokens=cfg.get("max_new_tokens", 256),
            bos_token_id=_BOS,
            eos_token_id=_EOS,
            pad_token_id=_PAD,
        )
        self._generation_config = GenerationConfig(**self._gen_kwargs)

        model_ref = model_path or HYMT_HF_ID
        self.tokenizer = AutoTokenizer.from_pretrained(model_ref)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _infer(self, item: EvalItem) -> tuple[str | None, str | None]:
        import torch

        text = item.resolve_input()
        prompt = (
            "Translate the following segment into English, "
            "without additional explanation.\n\n"
            f"{text}"
        )
        messages = [{"role": "user", "content": prompt}]
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model.generate(
                tokenized, generation_config=self._generation_config
            )
        generated = outputs[0][tokenized.shape[-1] :]
        translation = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return translation, None
