from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from scripts.prepare_incremental_v4 import PhotoProposal, normalize_rows, select_photos
from steel_ball.prelabel import Detection


def detection(x: float, *, uncertain: bool = False) -> Detection:
    return Detection(x, 55.0, 11.0, 10.0, 2.0, None, 0.0, uncertain)


def proposal(
    name: str,
    seconds: float,
    x: float,
    *,
    sharpness: float = 100.0,
    uncertain: bool = False,
) -> PhotoProposal:
    return PhotoProposal(
        Path(name),
        datetime(2026, 8, 2) + timedelta(seconds=seconds),
        1,
        detection(x, uncertain=uncertain),
        sharpness,
    )


def test_photo_selection_keeps_motion_time_endpoints_blur_and_uncertain() -> None:
    proposals = [
        proposal("first.jpg", 0.0, 100),
        proposal("duplicate.jpg", 0.1, 102),
        proposal("motion.jpg", 0.2, 112),
        proposal("elapsed.jpg", 1.5, 113),
        proposal("endpoint.jpg", 1.6, 450),
        proposal("blur.jpg", 1.7, 451, sharpness=1.0),
        proposal("uncertain.jpg", 1.8, 452, uncertain=True),
    ]
    selected = select_photos(proposals, 10.0, 1.0, 30.0, 0.01)
    assert [item.path.name for item in selected] == [
        "first.jpg",
        "motion.jpg",
        "elapsed.jpg",
        "endpoint.jpg",
        "blur.jpg",
        "uncertain.jpg",
    ]


def test_normalize_rows_preserves_pending_empty_sample() -> None:
    rows = normalize_rows(
        [
            {
                "relative_image": "images/train/empty.jpg",
                "split": "train",
                "status": "pending",
                "object_expected": "0",
            }
        ]
    )
    assert rows[0]["status"] == "pending"
    assert rows[0]["object_expected"] == "0"
    assert rows[0]["review_action"] == ""
