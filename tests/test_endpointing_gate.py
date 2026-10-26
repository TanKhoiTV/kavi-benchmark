"""Tests for the endpointing gate study, no model and no audio."""

from __future__ import annotations

from bench.endpointing_gate import (
    FINAL_MARKERS,
    NaiveBayes,
    auc,
    build_samples,
    features,
    normalise,
    recall_at_false_complete,
    rule_gate,
)


def test_normalise_lowercases_and_strips_punctuation() -> None:
    assert normalise("Anh giải thích lại chậm hơn được không?")[-1] == "không"
    assert normalise("  Bao lâu thì có đồ uống?  ")[0] == "bao"


def test_build_samples_yields_one_cut_per_word_boundary() -> None:
    samples = build_samples(["một hai ba"])
    # 3 tokens -> cuts after 1 and 2 words, plus the full sentence.
    assert len(samples) == 3
    assert sum(1 for s in samples if s.complete) == 1
    assert [s.text for s in samples if not s.complete] == ["một", "một hai"]


def test_single_token_sentence_is_skipped() -> None:
    assert build_samples(["xin"]) == []


def test_rule_gate_accepts_a_final_marker() -> None:
    assert rule_gate("Anh giải thích lại chậm hơn được không?") is True


def test_rule_gate_rejects_a_mid_sentence_prefix() -> None:
    assert rule_gate("Địa chỉ đúng là nhà bên cạnh, nhưng người giao") is False


def test_rule_gate_rejects_empty_text() -> None:
    assert rule_gate("   ") is False


def test_final_markers_include_the_common_question_particles() -> None:
    for marker in ("không", "chưa", "nhé", "nào", "đâu", "ạ"):
        assert marker in FINAL_MARKERS


def test_features_carry_the_last_token_and_a_length_bucket() -> None:
    feats = features("một hai ba bốn")
    assert "last:bốn" in feats
    assert "last2:ba_bốn" in feats
    assert any(f.startswith("len:") for f in feats)


def test_recall_at_zero_budget_never_accepts_a_prefix() -> None:
    scores = [(0.9, False), (0.1, True)]  # deliberately inverted
    recall, _ = recall_at_false_complete(scores, 0.0)
    assert recall == 0.0


def test_recall_at_full_budget_accepts_everything() -> None:
    scores = [(0.9, True), (0.1, False)]
    recall, threshold = recall_at_false_complete(scores, 1.0)
    assert recall == 1.0
    # At equal recall the stricter threshold wins, so a real gate is not made
    # permissive for free by a budget that happens to allow it.
    assert threshold == 0.9


def test_auc_is_one_for_perfect_separation() -> None:
    assert auc([(0.9, True), (0.1, False)]) == 1.0
    assert auc([(0.1, True), (0.9, False)]) == 0.0


def test_naive_bayes_prefers_the_full_sentence_it_was_trained_on() -> None:
    # The gate must learn from text alone that a dangling prefix is unfinished.
    samples = build_samples(
        [
            "anh giải thích lại chậm hơn được không",
            "bao lâu thì có đồ uống rồi",
            "phòng đang dọn khoảng hai mươi phút nữa là xong",
        ]
    )
    model = NaiveBayes()
    model.fit(samples)
    full = "phòng đang dọn khoảng hai mươi phút nữa là xong"
    prefix = "phòng đang dọn"
    assert model.log_odds(full) > model.log_odds(prefix)
