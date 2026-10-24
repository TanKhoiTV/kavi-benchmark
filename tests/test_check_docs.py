"""Tests for the doc shape check (``scripts/check_docs.py``).

Each test pins one rule, so a rule that silently stops firing fails here
instead of in review.
"""

from __future__ import annotations

from scripts.check_docs import body_lines, check_text, normalise, tracked_docs


def doc(*lines: str) -> str:
    body = list(lines)
    while body and not body[-1]:
        body.pop()
    return "\n".join(body) + "\n"


def test_clean_doc_reports_nothing() -> None:
    text = doc("## Description", "", "One paragraph on one line.", "", "- one", "- two")
    assert check_text(text) == []


def test_hard_wrapped_paragraph_is_reported() -> None:
    assert "1: hard-wrapped paragraph" in check_text(
        doc("A paragraph that was", "wrapped by hand.")
    )


def test_wrapping_is_joined() -> None:
    text = doc("A paragraph that was", "wrapped by hand.")
    assert normalise(body_lines(text)) == ["A paragraph that was wrapped by hand."]


def test_blank_line_inside_a_list_is_reported() -> None:
    assert "2: blank line inside a list" in check_text(doc("- one", "", "- two"))


def test_two_blank_lines_are_reported() -> None:
    assert "3: two blank lines in a row" in check_text(doc("First.", "", "", "Second."))


def test_heading_needs_a_blank_line_after_it() -> None:
    assert "1: no blank line after heading" in check_text(
        doc("## Description", "Text right after.")
    )


def test_missing_final_newline_is_reported() -> None:
    assert "missing final newline" in check_text("Text with no newline.")


def test_trailing_whitespace_is_reported() -> None:
    assert "1: trailing whitespace" in check_text(doc("Line with a trailing space. "))


def test_code_fence_content_is_left_alone() -> None:
    text = doc("```text", "a line,", "another line", "```")
    assert check_text(text) == []


def test_adrs_and_archive_are_out_of_scope() -> None:
    paths = [p.as_posix() for p in tracked_docs()]
    assert not [p for p in paths if "/ADR-" in p]
    assert not [p for p in paths if "/archive/" in p]
    assert len(paths) > 3
