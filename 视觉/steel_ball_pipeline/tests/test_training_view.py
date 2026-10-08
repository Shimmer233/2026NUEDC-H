from pathlib import Path

from steel_ball.training_view import (
    _eligible_for_synthetic_blur,
    _expanded_motion_label,
    _is_hard_example,
)


def test_motion_label_expands_with_horizontal_blur() -> None:
    label = "0 0.50000000 0.50000000 0.10000000 0.20000000\n"
    expanded = _expanded_motion_label(label, 100, 50, 11, 0.0).split()
    assert float(expanded[3]) == 0.2
    assert float(expanded[4]) == 0.2


def test_motion_label_preserves_empty_negative() -> None:
    assert _expanded_motion_label("", 100, 50, 31, 0.0) == ""


def test_real_video_frames_are_not_blurred_again() -> None:
    assert _eligible_for_synthetic_blur(Path("base__frame.jpg"))
    assert not _eligible_for_synthetic_blur(Path("adapt__fast_000123.jpg"))
    assert not _eligible_for_synthetic_blur(Path("newvideo__fast_000123.jpg"))


def test_motion_endpoint_and_redrawn_samples_are_hard_examples() -> None:
    label = "0 0.50000000 0.50000000 0.10000000 0.20000000\n"
    assert _is_hard_example(
        Path("newvideo__fast_000123.jpg"), label, {"source_kind": "fast"}
    )
    assert _is_hard_example(Path("newphoto__frame.jpg"), label, {"review_action": "redrawn"})
    assert _is_hard_example(
        Path("newphoto__endpoint.jpg"), "0 0.05000000 0.5 0.1 0.2\n", {}
    )
    assert not _is_hard_example(Path("newphoto__middle.jpg"), label, {})
