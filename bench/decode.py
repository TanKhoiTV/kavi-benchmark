"""Per-utterance decode-length guard.

CTranslate2's greedy decode loops on longer Vietnamese input: with a flat
``max_decoding_length=256`` cap, 12 of 50 VI->EN FLEURS sentences ran to the cap
and produced repetition ("... sch le el el el ..."), and those items set the
latency p95 on device (3.0 s against a 0.70 s median).

The fix is to size the cap from the input instead of the model. Measured on 347
VI->EN pairs with the pinned Opus-MT tokenizers, the target/source piece ratio is

     p50 0.81 | p95 1.08 | p99 1.17 | max 1.28

so a 1.5x multiplier sits ~17% above the longest legitimate output. The floor
keeps short utterances from being clipped by rounding, and the ceiling keeps a
pathological input from spending minutes on one sentence.

This is a guard, not a tuning knob: it bounds the damage, and the reported
latency tail is expected to shrink without touching beam width.
"""

from __future__ import annotations

import math

#: Cap = RATIO x input pieces, clamped to [FLOOR, CEILING].
RATIO = 1.5
FLOOR = 48
CEILING = 256


def max_decoding_length(n_input_tokens: int) -> int:
    """Decode cap for one utterance, derived from its source token count."""
    if n_input_tokens <= 0:
        return FLOOR
    return max(FLOOR, min(CEILING, math.ceil(RATIO * n_input_tokens)))
