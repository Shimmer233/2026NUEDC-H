from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


SPLITS = ("train", "val", "test")
APPROVAL_FILE = ".review-approved.json"


def image_files(dataset_root: Path, splits: Iterable[str] = SPLITS) -> list[Path]:
    paths: list[Path] = []
    for split in splits:
        image_dir = dataset_root / "images" / split
        if image_dir.exists():
            paths.extend(sorted(image_dir.glob("*.jpg")))
    return paths


def label_for_image(dataset_root: Path, image_path: Path) -> Path:
    split = image_path.parent.name
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def validate_label(path: Path, require_one: bool = True) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return ["missing label"]
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    if require_one and len(lines) != 1:
        errors.append(f"expected one object, found {len(lines)}")
    if not require_one and lines:
        errors.append(f"expected no objects, found {len(lines)}")
    for index, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"line {index}: expected 5 fields, found {len(fields)}")
            continue
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"line {index}: non-numeric field")
            continue
        if class_id != 0:
            errors.append(f"line {index}: class id must be 0")
        if not all(0.0 < value <= 1.0 for value in values):
            errors.append(f"line {index}: normalized values must be in (0, 1]")
        x, y, width, height = values
        if x - width / 2 < 0 or x + width / 2 > 1:
            errors.append(f"line {index}: box exceeds horizontal image bounds")
        if y - height / 2 < 0 or y + height / 2 > 1:
            errors.append(f"line {index}: box exceeds vertical image bounds")
    return errors


def read_review_rows(dataset_root: Path) -> dict[str, dict[str, str]]:
    status_path = dataset_root / "review" / "status.csv"
    if not status_path.exists():
        return {}
    with status_path.open("r", newline="", encoding="utf-8") as stream:
        return {row["relative_image"]: row for row in csv.DictReader(stream)}


def read_review_status(dataset_root: Path) -> dict[str, str]:
    return {relative: row["status"] for relative, row in read_review_rows(dataset_root).items()}


def labels_digest(dataset_root: Path) -> str:
    digest = hashlib.sha256()
    for image_path in image_files(dataset_root, ("train", "val")):
        relative = image_path.relative_to(dataset_root).as_posix()
        label_path = label_for_image(dataset_root, image_path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(label_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_approval(dataset_root: Path) -> tuple[bool, str]:
    approval_path = dataset_root / APPROVAL_FILE
    if not approval_path.exists():
        return False, f"approval marker is missing: {approval_path}"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    current_digest = labels_digest(dataset_root)
    if approval.get("labels_sha256") != current_digest:
        return False, "labels changed after review approval"
    return True, "approved"


def verify_split_review(dataset_root: Path, split: str) -> tuple[bool, str]:
    status_path = dataset_root / "review" / "status.csv"
    if not status_path.exists():
        return False, f"review status is missing: {status_path}"
    with status_path.open("r", newline="", encoding="utf-8") as stream:
        review_rows = {row["relative_image"]: row for row in csv.DictReader(stream)}
    images = image_files(dataset_root, (split,))
    if not images:
        return False, f"{split} split is empty"
    for image_path in images:
        relative = image_path.relative_to(dataset_root).as_posix()
        row = review_rows.get(relative)
        if row is None or row.get("status") != "accepted":
            return False, f"not reviewed: {relative}"
        row_expected = row.get("object_expected", "1") == "1"
        errors = validate_label(label_for_image(dataset_root, image_path), require_one=row_expected)
        if errors:
            return False, f"invalid reviewed label {relative}: {errors[0]}"
    return True, "reviewed"
