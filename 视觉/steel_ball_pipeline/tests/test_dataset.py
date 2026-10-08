from __future__ import annotations

from pathlib import Path

from steel_ball.dataset import validate_label


def test_validate_label_accepts_one_ball(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.1 0.2\n", encoding="ascii")
    assert validate_label(label) == []


def test_validate_label_rejects_confidence_column(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.1 0.2 0.9\n", encoding="ascii")
    errors = validate_label(label)
    assert any("expected 5 fields" in error for error in errors)


def test_validate_label_accepts_reviewed_empty_frame(tmp_path: Path) -> None:
    label = tmp_path / "empty.txt"
    label.write_text("", encoding="ascii")
    assert validate_label(label, require_one=False) == []


def test_validate_label_rejects_box_when_frame_is_marked_empty(tmp_path: Path) -> None:
    label = tmp_path / "not_empty.txt"
    label.write_text("0 0.5 0.5 0.1 0.2\n", encoding="ascii")
    assert any("expected no objects" in error for error in validate_label(label, require_one=False))
