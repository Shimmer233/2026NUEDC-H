from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.prelabel import TrackBallPrelabeler, yolo_line  # noqa: E402


FILENAME_PATTERN = re.compile(
    r"^WIN_(?P<date>\d{8})_(?P<hour>\d{2})_(?P<minute>\d{2})_"
    r"(?P<second>\d{2})_Pro(?: \((?P<sequence>\d+)\))?\.jpg$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and prelabel the isolated steel-ball dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "pipeline.yaml",
    )
    parser.add_argument("--force", action="store_true", help="Replace a previously generated dataset root.")
    return parser.parse_args()


def capture_key(path: Path) -> tuple[datetime, int, str]:
    match = FILENAME_PATTERN.match(path.name)
    if match is None:
        return datetime.max, 0, path.name
    timestamp = datetime.strptime(
        f"{match.group('date')} {match.group('hour')}:{match.group('minute')}:{match.group('second')}",
        "%Y%m%d %H:%M:%S",
    )
    sequence = int(match.group("sequence") or 1)
    return timestamp, sequence, path.name


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def read_crop(path: Path, roi: dict, grayscale: bool = False) -> np.ndarray:
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise RuntimeError(f"unable to read image: {path}")
    x, y = int(roi["x"]), int(roi["y"])
    width, height = int(roi["width"]), int(roi["height"])
    if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError(f"ROI exceeds image bounds for {path}: image shape={image.shape}")
    return image[y : y + height, x : x + width]


def split_for_timestamp(timestamp: datetime, config: dict) -> str:
    train_end = datetime.strptime(config["split"]["train_end"], "%Y-%m-%d %H:%M:%S")
    val_end = datetime.strptime(config["split"]["val_end"], "%Y-%m-%d %H:%M:%S")
    if timestamp <= train_end:
        return "train"
    if timestamp <= val_end:
        return "val"
    return "test"


def prepare_root(dataset_root: Path, force: bool) -> None:
    if dataset_root.exists() and any(dataset_root.iterdir()):
        if not force:
            raise RuntimeError(
                f"dataset root is not empty: {dataset_root}. Use --force only for generated data."
            )
        resolved = dataset_root.resolve()
        expected = Path(r"D:\University\NUEDC\datasets").resolve()
        if expected not in resolved.parents:
            raise RuntimeError(f"refusing to replace unexpected path: {resolved}")
        shutil.rmtree(resolved)
    for split in ("train", "val", "test"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels_auto" / split).mkdir(parents=True, exist_ok=True)
    (dataset_root / "review").mkdir(parents=True, exist_ok=True)
    (dataset_root / "manifests").mkdir(parents=True, exist_ok=True)
    (dataset_root / "calibration").mkdir(parents=True, exist_ok=True)


def make_background(paths: list[Path], roi: dict, sample_count: int) -> np.ndarray:
    indices = np.linspace(0, len(paths) - 1, min(sample_count, len(paths)), dtype=int)
    samples = [read_crop(paths[int(index)], roi, grayscale=True) for index in indices]
    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)


def write_dataset_yaml(dataset_root: Path) -> None:
    payload = {
        "path": dataset_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["ball"],
    }
    with (dataset_root / "steel_ball.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=False)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    source_dir = Path(config["source_images"])
    dataset_root = Path(config["dataset_root"])
    paths = sorted(source_dir.glob("*.jpg"), key=capture_key)
    if not paths:
        raise RuntimeError(f"no JPG images found in {source_dir}")
    unrecognized = [path.name for path in paths if FILENAME_PATTERN.match(path.name) is None]
    if unrecognized:
        raise RuntimeError(f"unrecognized image names: {unrecognized[:5]}")

    prepare_root(dataset_root, args.force)
    background = make_background(paths, config["roi"], int(config["prelabel"]["background_samples"]))
    cv2.imwrite(str(dataset_root / "calibration" / "median_background.png"), background)
    detector = TrackBallPrelabeler(background=background, **{
        key: value
        for key, value in config["prelabel"].items()
        if key != "background_samples"
    })

    manifest_rows: list[dict[str, str | float | bool | None]] = []
    counts = {"train": 0, "val": 0, "test": 0, "uncertain": 0}
    for index, source_path in enumerate(paths, start=1):
        timestamp, sequence, _ = capture_key(source_path)
        split = split_for_timestamp(timestamp, config)
        color_crop = read_crop(source_path, config["roi"], grayscale=False)
        gray_crop = cv2.cvtColor(color_crop, cv2.COLOR_BGR2GRAY)
        detection = detector.locate(gray_crop)
        destination_image = dataset_root / "images" / split / source_path.name
        destination_label = dataset_root / "labels" / split / f"{source_path.stem}.txt"
        auto_label = dataset_root / "labels_auto" / split / f"{source_path.stem}.txt"
        if not cv2.imwrite(str(destination_image), color_crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"unable to write image: {destination_image}")
        label_text = yolo_line(detection, color_crop.shape[1], color_crop.shape[0])
        destination_label.write_text(label_text, encoding="ascii")
        auto_label.write_text(label_text, encoding="ascii")
        relative_image = destination_image.relative_to(dataset_root).as_posix()
        manifest_rows.append({
            "index": index,
            "relative_image": relative_image,
            "source_image": str(source_path),
            "split": split,
            "timestamp": timestamp.isoformat(sep=" "),
            "sequence": sequence,
            "status": "pending",
            "object_expected": "1",
            **detection.to_dict(),
        })
        counts[split] += 1
        counts["uncertain"] += int(detection.uncertain)
        if index % 100 == 0 or index == len(paths):
            print(f"prelabelled {index}/{len(paths)}")

    fieldnames = list(manifest_rows[0].keys())
    for name in ("manifest.csv",):
        with (dataset_root / "manifests" / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
    review_fields = [
        "relative_image",
        "split",
        "status",
        "uncertain",
        "score",
        "peak_gap",
        "object_expected",
    ]
    with (dataset_root / "review" / "status.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in review_fields} for row in manifest_rows)

    write_dataset_yaml(dataset_root)
    metadata = {
        "source_images": str(source_dir),
        "source_count": len(paths),
        "source_mutated": False,
        "roi": config["roi"],
        "counts": counts,
        "config": str(args.config.resolve()),
    }
    (dataset_root / "manifests" / "build.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"Review every label before training: {dataset_root / 'review' / 'status.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
