from __future__ import annotations

import pytest

from steel_ball.session_zero import (
    create_session_zero_marker,
    invalidate_session_zero_marker,
    validate_session_zero_marker,
)


def validate(path, *, current_count: int = 0, boot_id: str = "boot-a"):
    return validate_session_zero_marker(
        path,
        address=1,
        zero_count=0,
        tolerance_counts=20,
        min_count=-200,
        max_count=200,
        current_count=current_count,
        boot_id=boot_id,
    )


def test_session_zero_marker_is_bound_to_boot_and_live_count(tmp_path) -> None:
    path = tmp_path / "session-zero.json"
    create_session_zero_marker(
        path,
        address=1,
        zero_count=0,
        tolerance_counts=20,
        boot_id="boot-a",
    )
    assert validate(path).boot_id == "boot-a"
    with pytest.raises(RuntimeError, match="previous system boot"):
        validate(path, boot_id="boot-b")
    with pytest.raises(RuntimeError, match="changed by 21"):
        validate(path, current_count=21)
    with pytest.raises(RuntimeError, match="outside session soft limits"):
        validate(path, current_count=35921)


def test_session_zero_marker_can_be_explicitly_invalidated(tmp_path) -> None:
    path = tmp_path / "session-zero.json"
    create_session_zero_marker(
        path,
        address=1,
        zero_count=0,
        tolerance_counts=20,
        boot_id="boot-a",
    )
    invalidate_session_zero_marker(path)
    assert not path.exists()
    with pytest.raises(RuntimeError, match="session zero is missing"):
        validate(path)
