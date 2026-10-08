from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.dataset import verify_split_review  # noqa: E402


def read_boxes(path: Path, width: int, height: int, confidence: float) -> list[np.ndarray]:
    if not path.exists():
        return []
    boxes: list[np.ndarray] = []
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) not in (5, 6):
            raise ValueError(f"invalid label line in {path}: {line}")
        if int(fields[0]) != 0:
            continue
        if len(fields) == 6 and float(fields[5]) < confidence:
            continue
        center_x, center_y, box_width, box_height = map(float, fields[1:5])
        boxes.append(
            np.asarray(
                [
                    (center_x - box_width / 2) * width,
                    (center_y - box_height / 2) * height,
                    (center_x + box_width / 2) * width,
                    (center_y + box_height / 2) * height,
                ]
            )
        )
    return boxes


def iou(first: np.ndarray, second: np.ndarray) -> float:
    upper_left = np.maximum(first[:2], second[:2])
    lower_right = np.minimum(first[2:], second[2:])
    intersection_size = np.maximum(0.0, lower_right - upper_left)
    intersection = float(np.prod(intersection_size))
    first_area = float(np.prod(np.maximum(0.0, first[2:] - first[:2])))
    second_area = float(np.prod(np.maximum(0.0, second[2:] - second[:2])))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute frame recall, false positives and hard-subset recall.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image_paths = sorted((args.dataset / "images" / args.split).glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"empty split: {args.split}")
    reviewed, reason = verify_split_review(args.dataset, args.split)
    if not reviewed:
        raise RuntimeError(f"analysis blocked: {reason}")
    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"unable to read {image_path}")
        height, width = image.shape
        ground_truth = read_boxes(
            args.dataset / "labels" / args.split / f"{image_path.stem}.txt",
            width,
            height,
            0.0,
        )
        predictions = read_boxes(
            args.predictions / f"{image_path.stem}.txt", width, height, args.confidence
        )
        matched = bool(
            ground_truth
            and any(iou(ground_truth[0], prediction) >= args.iou for prediction in predictions)
        )
        center_x = (
            float((ground_truth[0][0] + ground_truth[0][2]) / (2 * width))
            if ground_truth
            else None
        )
        rows.append(
            {
                "has_ball": bool(ground_truth),
                "matched": matched,
                "false_positive": not ground_truth and bool(predictions),
                "brightness": float(image.mean()),
                "sharpness": float(cv2.Laplacian(image, cv2.CV_64F).var()),
                "reflection": float(np.mean(image >= 245)),
                "center_x": center_x,
            }
        )
    ball_rows = [row for row in rows if row["has_ball"]]
    brightness_cut = np.quantile([row["brightness"] for row in ball_rows], 0.2) if ball_rows else 0
    sharpness_cut = np.quantile([row["sharpness"] for row in ball_rows], 0.2) if ball_rows else 0
    reflection_cut = np.quantile([row["reflection"] for row in ball_rows], 0.8) if ball_rows else 0
    subsets = {
        "all_ball_frames": ball_rows,
        "dark_frames": [row for row in ball_rows if row["brightness"] <= brightness_cut],
        "strong_reflection": [row for row in ball_rows if row["reflection"] >= reflection_cut],
        "track_ends": [
            row for row in ball_rows if row["center_x"] < 0.15 or row["center_x"] > 0.85
        ],
        "high_blur": [row for row in ball_rows if row["sharpness"] <= sharpness_cut],
    }
    subset_report = {
        name: {
            "frames": len(subset),
            "matched": sum(bool(row["matched"]) for row in subset),
            "recall": (
                sum(bool(row["matched"]) for row in subset) / len(subset) if subset else None
            ),
        }
        for name, subset in subsets.items()
    }
    longest_miss = 0
    current_miss = 0
    for row in rows:
        current_miss = current_miss + 1 if row["has_ball"] and not row["matched"] else 0
        longest_miss = max(longest_miss, current_miss)
    empty_count = sum(not row["has_ball"] for row in rows)
    false_positives = sum(bool(row["false_positive"]) for row in rows)
    report = {
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "confidence_threshold": args.confidence,
        "iou_threshold": args.iou,
        "subsets": subset_report,
        "longest_consecutive_misses": longest_miss,
        "empty_frames": empty_count,
        "false_positive_frames": false_positives,
        "false_positives_per_1000_empty_frames": (
            false_positives * 1000 / empty_count if empty_count else None
        ),
        "passes_ball_recall_99_percent": bool(
            ball_rows and subset_report["all_ball_frames"]["recall"] >= 0.99
        ),
        "passes_max_3_consecutive_misses": longest_miss <= 3,
        "passes_empty_fp_rate": bool(empty_count and false_positives * 1000 / empty_count <= 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
