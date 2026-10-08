from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.inspect_prelabels import render_item, select_rows
from scripts.review_labels import box_text


def test_select_rows_defaults_to_new_media() -> None:
    rows = [
        {"source_dataset": "base_v3", "status": "accepted", "uncertain": "False"},
        {"source_dataset": "new_media", "status": "pending", "uncertain": "False"},
        {"source_dataset": "new_media", "status": "pending", "uncertain": "True"},
    ]

    selected = select_rows(
        rows,
        include_old=False,
        pending_only=True,
        uncertain_only=True,
    )

    assert selected == [rows[2]]


def test_render_item_keeps_complete_image_and_adds_separate_detail_panel(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "labels_auto" / "train").mkdir(parents=True)
    image_path = dataset / "images" / "train" / "sample.jpg"
    image = np.full((110, 470, 3), 160, dtype=np.uint8)
    image[:, 0] = (10, 20, 30)
    image[:, -1] = (40, 50, 60)
    assert cv2.imwrite(str(image_path), image)
    (dataset / "labels_auto" / "train" / "sample.txt").write_text(
        box_text((210, 35, 230, 55), 470, 110), encoding="ascii"
    )
    row = {
        "relative_image": "images/train/sample.jpg",
        "split": "train",
        "status": "pending",
        "uncertain": "False",
    }

    canvas = render_item(dataset, row, 0, 1, scale=2)

    assert canvas.shape[0] == 64 + 110 * 2
    assert canvas.shape[1] == 470 * 2 + 240
    # Both source edges remain present in the full-track pane.
    assert tuple(canvas[64 + 100, 1]) != tuple(canvas[64 + 100, 470 * 2 - 2])
