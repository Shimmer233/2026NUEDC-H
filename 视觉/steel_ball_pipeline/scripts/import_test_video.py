from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.prelabel import TrackBallPrelabeler, yolo_line  # noqa: E402


def crop_and_resize(frame: np.ndarray, roi: dict, output_size: tuple[int, int] | None) -> np.ndarray:
    x, y, width, height = (int(roi[key]) for key in ("x", "y", "width", "height"))
    crop = frame[y : y + height, x : x + width]
    if crop.shape[:2] != (height, width):
        raise ValueError(f"ROI {roi} exceeds video frame shape {frame.shape}")
    if output_size is not None:
        crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
    return crop


def sampled_background(
    video_path: Path,
    roi: dict,
    output_size: tuple[int, int] | None,
    sample_count: int = 300,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"video contains no frames: {video_path}")
    samples: list[np.ndarray] = []
    for frame_index in np.linspace(0, frame_count - 1, min(frame_count, sample_count), dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue
        crop = crop_and_resize(frame, roi, output_size)
        samples.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    capture.release()
    if not samples:
        raise RuntimeError("unable to sample video background")
    return np.median(np.stack(samples), axis=0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import an independent video into the test split.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--kind", choices=("ball", "empty"), required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--confirm-empty", action="store_true")
    parser.add_argument(
        "--roi",
        help="Override source ROI as x,y,width,height (for example 170,190,1240,290).",
    )
    parser.add_argument(
        "--output-size",
        help="Resize every crop to width,height (for example 470,110).",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "pipeline.yaml")
    args = parser.parse_args()
    if args.kind == "empty" and not args.confirm_empty:
        raise RuntimeError("empty videos require --confirm-empty after visually verifying the full recording")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = Path(config["dataset_root"])
    roi = dict(config["roi"])
    if args.roi:
        x, y, width, height = (int(value) for value in args.roi.split(","))
        roi = {"x": x, "y": y, "width": width, "height": height}
    output_size = None
    if args.output_size:
        output_size = tuple(int(value) for value in args.output_size.split(","))
        if len(output_size) != 2:
            raise ValueError("--output-size must be width,height")
    image_dir = dataset / "images" / "test"
    label_dir = dataset / "labels" / "test"
    auto_dir = dataset / "labels_auto" / "test"
    for directory in (image_dir, label_dir, auto_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if list(image_dir.glob(f"{args.prefix}_*.jpg")):
        raise RuntimeError(f"test prefix already exists: {args.prefix}")
    detector = None
    if args.kind == "ball":
        background = sampled_background(args.video, roi, output_size)
        detector = TrackBallPrelabeler(
            background=background,
            **{
                key: value
                for key, value in config["prelabel"].items()
                if key != "background_samples"
            },
        )
    capture = cv2.VideoCapture(str(args.video))
    rows: list[dict[str, str]] = []
    frame_index = 0
    written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % args.stride:
            frame_index += 1
            continue
        crop = crop_and_resize(frame, roi, output_size)
        height, width = crop.shape[:2]
        stem = f"{args.prefix}_{frame_index:06d}"
        image_path = image_dir / f"{stem}.jpg"
        label_path = label_dir / f"{stem}.txt"
        auto_path = auto_dir / f"{stem}.txt"
        cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if detector is None:
            label_text = ""
            uncertain, score, peak_gap = "False", "", ""
            status = "accepted"
            expected = "0"
        else:
            detection = detector.locate(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            label_text = yolo_line(detection, width, height)
            uncertain = str(detection.uncertain)
            score = str(detection.score)
            peak_gap = str(detection.peak_gap)
            status = "pending"
            expected = "1"
        label_path.write_text(label_text, encoding="ascii")
        auto_path.write_text(label_text, encoding="ascii")
        rows.append(
            {
                "relative_image": image_path.relative_to(dataset).as_posix(),
                "split": "test",
                "status": status,
                "uncertain": uncertain,
                "score": score,
                "peak_gap": peak_gap,
                "object_expected": expected,
            }
        )
        written += 1
        frame_index += 1
    capture.release()
    status_path = dataset / "review" / "status.csv"
    with status_path.open("r", newline="", encoding="utf-8") as stream:
        existing = list(csv.DictReader(stream))
    fields = [
        "relative_image",
        "split",
        "status",
        "uncertain",
        "score",
        "peak_gap",
        "object_expected",
    ]
    for row in existing:
        row.setdefault("object_expected", "1")
    with status_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + rows)
    print(f"Imported {written} test frames from {args.video}")
    if args.kind == "ball":
        print("Review the imported ball labels with scripts/review_labels.py --pending-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
