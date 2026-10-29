"""Tests for the register scorer's sheet reader - no model, no audio, no network.

COMET is deliberately not exercised: it needs a ~2 GB checkpoint, so the metric is
covered by the artifact it writes rather than by the unit suite.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from bench.register_eval import REGISTERS, Register, load_pairs


def _write_sheet(path: Path, header: tuple[str, str, str], rows: list[tuple]) -> Path:
    workbook = openpyxl.Workbook()
    # `Workbook.active` is Optional in openpyxl's own types; a fresh book always has
    # one sheet, and indexing keeps this typed without an assert.
    sheet = workbook.worksheets[0]
    sheet.append(list(header))
    for row in rows:
        sheet.append(list(row))
    workbook.save(path)
    return path


def test_load_pairs_reads_id_source_reference(tmp_path: Path) -> None:
    sheet = _write_sheet(
        tmp_path / "vi-en.xlsx",
        ("ID", "Tiếng Việt", "Tiếng Anh - Final"),
        [
            ("VE-0001", "Xin chào", "Hello"),
            ("VE-0002", "Cảm ơn", "Thank you"),
        ],
    )
    register = Register("vi-en", sheet, "vi", "en", source_note="test")

    pairs = load_pairs(register)

    assert [(p.item_id, p.source, p.reference) for p in pairs] == [
        ("VE-0001", "Xin chào", "Hello"),
        ("VE-0002", "Cảm ơn", "Thank you"),
    ]


def test_load_pairs_skips_rows_without_source_or_reference(tmp_path: Path) -> None:
    sheet = _write_sheet(
        tmp_path / "en-vi.xlsx",
        ("ID", "Tiếng Anh (normalized)", "Tiếng Việt (normalized)"),
        [
            ("EV-0001", "hello", "xin chào"),
            (None, None, None),
            ("EV-0002", "", "không nguồn"),
            ("EV-0003", "source only", None),
        ],
    )
    register = Register("en-vi", sheet, "en", "vi", source_note="test")

    pairs = load_pairs(register)

    assert [p.item_id for p in pairs] == ["EV-0001"]


def test_load_pairs_honours_limit(tmp_path: Path) -> None:
    sheet = _write_sheet(
        tmp_path / "vi-en.xlsx",
        ("ID", "Tiếng Việt", "Tiếng Anh - Final"),
        [(f"VE-{i:04d}", f"nguồn {i}", f"source {i}") for i in range(10)],
    )
    register = Register("vi-en", sheet, "vi", "en", source_note="test")

    assert len(load_pairs(register, limit=3)) == 3


def test_load_pairs_rejects_a_sheet_without_an_id_column(tmp_path: Path) -> None:
    sheet = _write_sheet(
        tmp_path / "bad.xlsx", ("No", "Tiếng Việt", "English"), [("x", "a", "b")]
    )
    register = Register("vi-en", sheet, "vi", "en", source_note="test")

    with pytest.raises(ValueError, match="'ID' first column"):
        load_pairs(register)


def test_load_pairs_names_the_missing_sheet(tmp_path: Path) -> None:
    register = Register("vi-en", tmp_path / "absent.xlsx", "vi", "en", source_note="t")

    with pytest.raises(FileNotFoundError, match="not redistributed"):
        load_pairs(register)


def test_registers_point_at_the_gitignored_calibration_tree() -> None:
    """The corpus is not redistributed, so the paths must stay inside eval_data/."""
    for direction, register in REGISTERS.items():
        # .../eval_data/organizer/<direction>/<sheet>.xlsx
        assert register.sheet.parent.name == direction, direction
        assert register.sheet.parent.parent.name == "organizer", direction
        assert register.sheet.parent.parent.parent.name == "eval_data", direction
        expected = "V_to_E" if direction == "vi-en" else "E_to_V"
        assert register.sheet.name == f"OneVoice_{expected}_samples_100.xlsx"
