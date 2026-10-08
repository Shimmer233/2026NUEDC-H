from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.review_labels as review_labels
from scripts.review_labels import (
    Reviewer,
    box_text,
    first_pending_index,
    read_status,
    review_labels_digest,
)


FIELDS = [
    "relative_image",
    "split",
    "status",
    "uncertain",
    "object_expected",
]
AUTO_BOX = (20, 30, 40, 50)


def make_dataset(tmp_path: Path, statuses: list[str]) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    for relative in ("images/train", "labels/train", "labels_auto/train", "review"):
        (dataset / relative).mkdir(parents=True, exist_ok=True)

    rows = []
    for index, status in enumerate(statuses):
        stem = f"sample_{index}"
        image_path = dataset / "images" / "train" / f"{stem}.jpg"
        assert cv2.imwrite(str(image_path), np.full((80, 100, 3), 180, dtype=np.uint8))
        label = box_text(AUTO_BOX, 100, 80)
        (dataset / "labels" / "train" / f"{stem}.txt").write_text(label, encoding="ascii")
        (dataset / "labels_auto" / "train" / f"{stem}.txt").write_text(label, encoding="ascii")
        rows.append(
            {
                "relative_image": f"images/train/{stem}.jpg",
                "split": "train",
                "status": status,
                "uncertain": "False",
                "object_expected": "1",
            }
        )

    status_path = dataset / "review" / "status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return dataset, status_path


def load_reviewer(dataset: Path, status_path: Path) -> Reviewer:
    rows, fields = read_status(status_path)
    reviewer = Reviewer(dataset, rows, fields, status_path=status_path)
    reviewer.load()
    return reviewer


def test_resume_starts_at_first_pending_without_reordering_rows(tmp_path: Path) -> None:
    dataset, status_path = make_dataset(tmp_path, ["accepted", "pending", "pending"])
    reviewer = load_reviewer(dataset, status_path)

    assert first_pending_index(reviewer.rows) == 1
    assert reviewer.index == 1
    assert [row["relative_image"] for row in reviewer.rows] == [
        "images/train/sample_0.jpg",
        "images/train/sample_1.jpg",
        "images/train/sample_2.jpg",
    ]


def test_redraw_stays_pending_until_explicit_accept(tmp_path: Path) -> None:
    dataset, status_path = make_dataset(tmp_path, ["pending"])
    reviewer = load_reviewer(dataset, status_path)

    reviewer.mouse(cv2.EVENT_LBUTTONDOWN, 20, 24, 0, None)
    reviewer.mouse(cv2.EVENT_LBUTTONUP, 100, 120, 0, None)

    assert reviewer.current_box == (10, 12, 50, 60)
    assert reviewer.rows[0]["status"] == "pending"
    persisted, _ = read_status(status_path)
    assert persisted[0]["status"] == "pending"

    report_path = reviewer.accept_current_box()
    persisted, _ = read_status(status_path)
    assert persisted[0]["status"] == "accepted"
    assert persisted[0]["object_expected"] == "1"
    assert persisted[0]["review_action"] == "redrawn"
    assert report_path == dataset / "review" / "completion_report.json"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["total"] == 1
    assert report["ball"] == 1
    assert report["empty"] == 0
    assert report["redrawn"] == 1
    assert report["accepted_auto"] == 0
    assert report["labels_sha256"] == review_labels_digest(dataset, reviewer.rows)


def test_mark_empty_requires_n_action_and_writes_empty_label(tmp_path: Path) -> None:
    dataset, status_path = make_dataset(tmp_path, ["pending"])
    reviewer = load_reviewer(dataset, status_path)
    reviewer.current_box = None

    with pytest.raises(ValueError, match="press N"):
        reviewer.accept_current_box()
    assert reviewer.rows[0]["status"] == "pending"

    reviewer.mark_current_empty()
    persisted, _ = read_status(status_path)
    label_path = dataset / "labels" / "train" / "sample_0.txt"
    assert label_path.read_text(encoding="ascii") == ""
    assert persisted[0]["status"] == "accepted"
    assert persisted[0]["object_expected"] == "0"
    assert persisted[0]["review_action"] == "empty"

    report = json.loads((dataset / "review" / "completion_report.json").read_text(encoding="utf-8"))
    assert report["ball"] == 0
    assert report["empty"] == 1


def test_reset_restores_auto_box_but_still_requires_confirmation(tmp_path: Path) -> None:
    dataset, status_path = make_dataset(tmp_path, ["pending"])
    reviewer = load_reviewer(dataset, status_path)
    reviewer.current_box = (5, 5, 25, 25)
    reviewer.was_redrawn = True

    reviewer.reset_current()

    assert reviewer.current_box == AUTO_BOX
    assert reviewer.was_redrawn is False
    assert reviewer.rows[0]["status"] == "pending"
    assert not (dataset / "review" / "completion_report.json").exists()

    reviewer.accept_current_box()
    persisted, _ = read_status(status_path)
    assert persisted[0]["review_action"] == "accepted_auto"


def test_completion_report_is_deferred_until_every_row_is_accepted(tmp_path: Path) -> None:
    dataset, status_path = make_dataset(tmp_path, ["pending", "pending"])
    reviewer = load_reviewer(dataset, status_path)

    assert reviewer.accept_current_box() is None
    assert not (dataset / "review" / "completion_report.json").exists()
    assert reviewer.move_to_next_pending()
    report_path = reviewer.mark_current_empty()

    assert report_path is not None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["total"] == 2
    assert report["ball"] == 1
    assert report["empty"] == 1
    assert report["redrawn"] == 0
    assert report["accepted_auto"] == 1


def test_box_is_clamped_and_must_remain_at_least_four_pixels() -> None:
    assert box_text((-5, -3, 10, 12), 100, 80).startswith("0 ")
    with pytest.raises(ValueError, match="at least 4x4"):
        box_text((101, 10, 120, 30), 100, 80)


def test_failed_status_save_does_not_mark_sample_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, status_path = make_dataset(tmp_path, ["pending"])
    reviewer = load_reviewer(dataset, status_path)

    def fail_save(*_args, **_kwargs) -> None:
        raise OSError("simulated status replace failure")

    monkeypatch.setattr(review_labels, "save_status", fail_save)
    with pytest.raises(OSError, match="simulated"):
        reviewer.accept_current_box()

    assert reviewer.rows[0]["status"] == "pending"
    persisted, _ = read_status(status_path)
    assert persisted[0]["status"] == "pending"


def test_render_keeps_full_image_visible_and_uses_separate_information_bar(tmp_path: Path) -> None:
    dataset, status_path = make_dataset(tmp_path, ["pending"])
    reviewer = load_reviewer(dataset, status_path)

    rendered = reviewer.render()

    assert rendered.shape[:2] == (80 * 2 + review_labels.INFO_HEIGHT, 100 * 2)
