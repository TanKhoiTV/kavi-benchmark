"""Can a text gate tell a finished Vietnamese sentence from a truncated prefix?

The pause measurement (`pause-distribution.md`) showed the silence timeout is boxed in:
lowering it truncates speech, raising it breaks the latency gate. The speech-dialogue
literature's answer is to stop treating silence duration as evidence of completion and let
the transcript decide (kavi-android#56). This tests whether that evidence exists in the text
at all, before anyone builds a model or a device pipeline around it.

Method: every Vietnamese sentence is cut at every word boundary. Each cut is one sample:
  * truncated prefix  -> the gate must say "incomplete" (saying otherwise truncates speech);
  * the full sentence -> the gate must say "complete"  (saying otherwise costs latency).

The asymmetry drives the whole evaluation: a false "complete" loses content, a false
"incomplete" only waits longer. So the headline number is recall on true ends at a fixed
false-complete budget, not accuracy.

Two gates are compared, both computable from a hypothesis alone (no reference, no total
length, nothing the device would not have):
  * final-particle rule: complete iff the last token is a Vietnamese sentence-final marker;
  * naive Bayes over token unigrams, bigrams and a length bucket.

Cross-validation holds out whole sentences, not samples, so a gate cannot memorise the
sentence it is being scored on.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Sentence-final markers in Vietnamese. Questions frequently end in không/chưa/gì/nào/đâu/
# chứ/ạ/nhé; imperatives and confirmations end in nhé/nhá/đi/ạ/rồi. This list is the whole
# content of the rule baseline, so the comparison against the learned gate is fair.
FINAL_MARKERS = frozenset(
    {
        "không",
        "chưa",
        "gì",
        "nào",
        "đâu",
        "chứ",
        "ạ",
        "nhé",
        "nhá",
        "đi",
        "rồi",
        "vậy",
        "thế",
        "hả",
        "sao",
    }
)

# Function words that almost never end a Vietnamese sentence.
NON_FINAL_WORDS = frozenset(
    {
        "và",
        "hoặc",
        "nhưng",
        "vì",
        "nếu",
        "thì",
        "để",
        "của",
        "cho",
        "với",
        "trong",
        "ngoài",
        "trên",
        "dưới",
        "khi",
        "sau",
        "trước",
        "là",
        "có",
        "đang",
        "sẽ",
        "đã",
        "một",
        "những",
        "các",
    }
)


@dataclass(frozen=True)
class Sample:
    sentence: int
    text: str
    complete: bool


def normalise(text: str) -> list[str]:
    """Tokens for scoring: lowercase, punctuation stripped, whitespace split."""
    cleaned = text.strip().lower()
    for mark in ".!?,;:":
        cleaned = cleaned.replace(mark, " ")
    return [t for t in cleaned.split() if t]


def build_samples(sentences: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for index, sentence in enumerate(sentences):
        tokens = sentence.split()
        if len(tokens) < 2:
            continue
        for cut in range(1, len(tokens)):
            samples.append(Sample(index, " ".join(tokens[:cut]), complete=False))
        samples.append(Sample(index, sentence, complete=True))
    return samples


def length_bucket(n: int) -> str:
    if n <= 2:
        return "len:<=2"
    if n <= 4:
        return "len:3-4"
    if n <= 7:
        return "len:5-7"
    if n <= 11:
        return "len:8-11"
    return "len:>=12"


def features(text: str) -> list[str]:
    tokens = normalise(text)
    if not tokens:
        return ["empty"]
    feats = ["bias", length_bucket(len(tokens)), f"last:{tokens[-1]}"]
    if len(tokens) >= 2:
        feats.append(f"last2:{tokens[-2]}_{tokens[-1]}")
    feats.extend(f"tok:{t}" for t in tokens)
    return feats


def rule_gate(text: str) -> bool:
    tokens = normalise(text)
    if not tokens:
        return False
    return tokens[-1] in FINAL_MARKERS and tokens[-1] not in NON_FINAL_WORDS


class NaiveBayes:
    """Multinomial naive Bayes over string features, Laplace-smoothed."""

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.counts: dict[bool, Counter] = {True: Counter(), False: Counter()}
        self.totals: dict[bool, int] = {True: 0, False: 0}
        self.docs: dict[bool, int] = {True: 0, False: 0}
        self.vocab: set[str] = set()

    def fit(self, samples: list[Sample]) -> None:
        for sample in samples:
            feats = features(sample.text)
            self.counts[sample.complete].update(feats)
            self.totals[sample.complete] += len(feats)
            self.docs[sample.complete] += 1
            self.vocab.update(feats)

    def log_odds(self, text: str) -> float:
        """log P(complete) - log P(incomplete); positive means 'finished'."""
        size = len(self.vocab) + 1
        total_docs = self.docs[True] + self.docs[False]
        prior = math.log((self.docs[True] + 1) / (total_docs + 2)) - math.log(
            (self.docs[False] + 1) / (total_docs + 2)
        )
        score = prior
        for feat in features(text):
            for label, sign in ((True, 1.0), (False, -1.0)):
                numerator = self.counts[label][feat] + self.alpha
                denominator = self.totals[label] + self.alpha * size
                score += sign * math.log(numerator / denominator)
        return score


def recall_at_false_complete(
    scores: list[tuple[float, bool]], budget: float
) -> tuple[float, float]:
    """Best recall on true ends with false-complete rate <= budget.

    A false complete means a truncated prefix scored as finished, which loses speech, so
    the budget is a hard constraint and the threshold is chosen to respect it.
    """
    negatives = [s for s, complete in scores if not complete]
    positives = [s for s, complete in scores if complete]
    if not negatives or not positives:
        return 0.0, 0.0
    best_recall, best_threshold = 0.0, float("inf")
    # Candidate thresholds are the observed scores; sweeping them is exact for this data.
    for threshold in sorted(set(s for s, _ in scores)):
        false_complete = sum(1 for s in negatives if s >= threshold) / len(negatives)
        if false_complete > budget:
            continue
        recall = sum(1 for s in positives if s >= threshold) / len(positives)
        if recall > best_recall or (
            recall == best_recall and threshold > best_threshold
        ):
            best_recall, best_threshold = recall, threshold
    return best_recall, best_threshold


def auc(scores: list[tuple[float, bool]]) -> float:
    positives = [s for s, c in scores if c]
    negatives = [s for s, c in scores if not c]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def cross_validate(samples: list[Sample], folds: int, seed: int) -> dict:
    sentence_ids = sorted({s.sentence for s in samples})
    random.Random(seed).shuffle(sentence_ids)
    fold_of = {sentence: index % folds for index, sentence in enumerate(sentence_ids)}

    nb_scores: list[tuple[float, bool]] = []
    rule_scores: list[tuple[float, bool]] = []
    for fold in range(folds):
        train = [s for s in samples if fold_of[s.sentence] != fold]
        test = [s for s in samples if fold_of[s.sentence] == fold]
        model = NaiveBayes()
        model.fit(train)
        nb_scores.extend((model.log_odds(s.text), s.complete) for s in test)
        # The rule has nothing to train; scored on every fold for a like-for-like set.
        rule_scores.extend(
            (1.0 if rule_gate(s.text) else 0.0, s.complete) for s in test
        )

    return {
        "samples": len(samples),
        "positives": sum(1 for s in samples if s.complete),
        "negatives": sum(1 for s in samples if not s.complete),
        "naive_bayes": _score_block(nb_scores),
        "final_particle_rule": _score_block(rule_scores),
        "always_complete": {"note": "baseline: accepts every prefix as finished"},
    }


def _score_block(scores: list[tuple[float, bool]]) -> dict:
    block: dict = {"auc": round(auc(scores), 3)}
    for budget in (0.0, 0.01, 0.05, 0.10):
        recall, threshold = recall_at_false_complete(scores, budget)
        block[f"recall_at_false_complete_{int(budget * 100)}pct"] = round(recall, 3)
        block[f"threshold_{int(budget * 100)}pct"] = round(threshold, 3)
    return block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentences",
        required=True,
        help="UTF-8 text file, one Vietnamese sentence per line",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="bench-results/endpointing-gate.json")
    args = parser.parse_args(argv)

    sentences = [
        line.strip()
        for line in Path(args.sentences).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(sentences) < 10:
        raise SystemExit(f"need at least 10 sentences, got {len(sentences)}")

    samples = build_samples(sentences)
    result = cross_validate(samples, args.folds, args.seed)
    result["sentences"] = len(sentences)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"samples {result['samples']} ({result['positives']} complete, "
        f"{result['negatives']} truncated prefixes)"
    )
    for name in ("naive_bayes", "final_particle_rule"):
        block = result[name]
        print(
            f"{name:20s} AUC {block['auc']:.3f} | recall@0% "
            f"{block['recall_at_false_complete_0pct']:.3f} | @1% "
            f"{block['recall_at_false_complete_1pct']:.3f} | @5% "
            f"{block['recall_at_false_complete_5pct']:.3f} | @10% "
            f"{block['recall_at_false_complete_10pct']:.3f}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
