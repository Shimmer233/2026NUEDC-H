from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.import_test_video import crop_and_resize, sampled_background  # noqa: E402
from steel_ball.prelabel import Detection, TrackBallPrelabeler  # noqa: E402


DEFAULT_DATASET = Path(r"D:\University\NUEDC\datasets\steel_ball_video_adaptation_v2")
DEFAULT_NORMAL = Path(r"C:\Users\Light\Videos\正常往返.mp4")
DEFAULT_FAST = Path(r"C:\Users\Light\Videos\高速运动.mp4")
DEFAULT_EMPTY = Path(r"C:\Users\Light\Videos\空轨道.mp4")
STATUS_FIELDS = [
    "relative_image",
    "split",
    "status",
    "uncertain",
    "score",
    "peak_gap",
    "object_expected",
    "source_kind",
    "source_video",
    "source_frame",
    "source_time_ms",
    "prelabel_method",
    "box_width_px",
    "box_height_px",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_tuple(value: str, count: int, name: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if len(result) != count:
        raise argparse.ArgumentTypeError(f"{name} requires {count} comma-separated integers")
    return result


def box_to_yolo(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x0, y0, x1, y1 = box
    center_x = (x0 + x1) / 2.0 / width
    center_y = (y0 + y1) / 2.0 / height
    box_width = (x1 - x0) / width
    box_height = (y1 - y0) / height
    return f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}\n"


def motion_expanded_box(
    gray: np.ndarray,
    background: np.ndarray,
    detection: Detection,
    search_y_min: int,
    search_y_max: int,
) -> tuple[tuple[int, int, int, int], str, bool]:
    """Expand a circular proposal over a connected dark motion streak."""
    height, width = gray.shape
    aligned = np.clip(gray.astype(np.float32) + detection.exposure_offset, 0, 255)
    delta = np.maximum(background.astype(np.float32) - aligned, 0)
    delta = cv2.GaussianBlur(delta, (3, 3), 0.8)

    cx, cy = int(round(detection.center_x)), int(round(detection.center_y))
    local_x0, local_x1 = max(0, cx - 65), min(width, cx + 66)
    local_y0 = max(search_y_min, cy - 25)
    local_y1 = min(search_y_max, cy + 26)
    local = delta[local_y0:local_y1, local_x0:local_x1]
    local_peak = float(local.max()) if local.size else 0.0
    threshold = max(8.0, local_peak * 0.22)

    mask = np.zeros_like(gray, dtype=np.uint8)
    if local.size:
        mask[local_y0:local_y1, local_x0:local_x1] = (local >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for index in range(1, count):
        x, y, component_width, component_height, area = (int(v) for v in stats[index])
        if area < 10 or component_width < 3 or component_height < 3:
            continue
        if component_width > 100 or component_height > 45:
            continue
        component_x, component_y = (float(v) for v in centroids[index])
        distance = float(np.hypot(component_x - detection.center_x, component_y - detection.center_y))
        contains = x - 4 <= detection.center_x <= x + component_width + 4 and y - 4 <= detection.center_y <= y + component_height + 4
        if not contains and distance > 22:
            continue
        mean_delta = float(delta[y : y + component_height, x : x + component_width][mask[y : y + component_height, x : x + component_width] > 0].mean())
        score = mean_delta + min(area, 500) * 0.02 - distance
        candidates.append((score, (x, y, x + component_width, y + component_height)))

    if candidates:
        _, (x0, y0, x1, y1) = max(candidates, key=lambda item: item[0])
        x0, x1 = max(0, x0 - 4), min(width, x1 + 4)
        y0, y1 = max(0, y0 - 4), min(height, y1 + 4)
        component_width, component_height = x1 - x0, y1 - y0
        unusual = component_width < 10 or component_height < 8 or component_width > 75 or component_height > 35
        return (x0, y0, x1, y1), "motion_component", unusual

    radius = int(round(detection.radius))
    box = (
        max(0, cx - radius),
        max(0, cy - radius),
        min(width, cx + radius),
        min(height, cy + radius),
    )
    return box, "circular_fallback", True


def temporal_appearance_box(
    gray: np.ndarray,
    previous_gray: np.ndarray,
    next_gray: np.ndarray,
) -> tuple[tuple[int, int, int, int], float, float, bool]:
    """Find a ball/streak on the bright track using local darkness and temporal change."""
    current = gray.astype(np.float32)
    previous = previous_gray.astype(np.float32)
    following = next_gray.astype(np.float32)
    local_background = cv2.GaussianBlur(current, (0, 0), sigmaX=12, sigmaY=5)
    darkness = np.maximum(local_background - current, 0)
    darkness = cv2.GaussianBlur(darkness, (5, 5), 1.0)

    search = darkness.copy()
    search[:47] = -1
    search[63:] = -1
    search[:, :30] = -1
    search[:, 426:] = -1
    dilated = cv2.dilate(search, np.ones((9, 9), np.uint8))
    points_y, points_x = np.where((search == dilated) & (search > 8.0))
    raw = sorted(
        ((float(search[y, x]), int(x), int(y)) for y, x in zip(points_y, points_x)),
        reverse=True,
    )

    candidates: list[tuple[float, int, int, float]] = []
    for dark_score, center_x, center_y in raw:
        if any(
            (center_x - other_x) ** 2 + (center_y - other_y) ** 2 <= 12**2
            for _, other_x, other_y, _ in candidates
        ):
            continue
        x0, x1 = max(0, center_x - 8), min(gray.shape[1], center_x + 9)
        y0, y1 = max(0, center_y - 8), min(gray.shape[0], center_y + 9)
        temporal_change = float(
            (
                np.abs(current[y0:y1, x0:x1] - previous[y0:y1, x0:x1])
                + np.abs(following[y0:y1, x0:x1] - current[y0:y1, x0:x1])
            ).mean()
            / 2.0
        )
        endpoint_penalty = 60.0 if center_x > 410 else (25.0 if center_x < 45 else 0.0)
        combined_score = dark_score + 2.0 * temporal_change - endpoint_penalty
        candidates.append((combined_score, center_x, center_y, dark_score))
        if len(candidates) >= 15:
            break

    if not candidates:
        return (208, 40, 232, 70), 0.0, 0.0, True

    candidates.sort(reverse=True)
    best_score, center_x, center_y, dark_score = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    peak_gap = best_score - second_score

    column_score = np.max(darkness[47:63], axis=0)
    threshold = max(8.0, dark_score * 0.28)
    left = right = center_x
    gap = 0
    for x in range(center_x - 1, max(29, center_x - 65), -1):
        if column_score[x] >= threshold:
            left, gap = x, 0
        else:
            gap += 1
            if gap >= 3:
                break
    gap = 0
    for x in range(center_x + 1, min(426, center_x + 66)):
        if column_score[x] >= threshold:
            right, gap = x, 0
        else:
            gap += 1
            if gap >= 3:
                break

    box = (
        max(0, left - 6),
        max(0, center_y - 15),
        min(gray.shape[1], right + 7),
        min(gray.shape[0], center_y + 16),
    )
    width = box[2] - box[0]
    unusual = best_score < 80.0 or center_x < 45 or center_x > 410 or width > 65
    return box, best_score, peak_gap, unusual


def refine_sequence_labels(dataset: Path, rows: list[dict[str, str]]) -> None:
    """Second pass: use neighboring sampled frames to reject static track hardware."""
    for kind in ("normal", "fast"):
        group = sorted(
            (row for row in rows if row["source_kind"] == kind),
            key=lambda row: int(row["source_frame"]),
        )
        grays = [
            cv2.imread(str(dataset / row["relative_image"]), cv2.IMREAD_GRAYSCALE)
            for row in group
        ]
        if any(gray is None for gray in grays):
            raise RuntimeError(f"unable to read one or more generated {kind} images")
        for index, (row, gray) in enumerate(zip(group, grays)):
            previous = grays[max(0, index - 1)]
            following = grays[min(len(grays) - 1, index + 1)]
            box, score, peak_gap, uncertain = temporal_appearance_box(gray, previous, following)
            image_path = dataset / row["relative_image"]
            label_text = box_to_yolo(box, gray.shape[1], gray.shape[0])
            for label_root in ("labels", "labels_auto"):
                (dataset / label_root / row["split"] / f"{image_path.stem}.txt").write_text(
                    label_text,
                    encoding="ascii",
                )
            row.update(
                {
                    "uncertain": str(uncertain),
                    "score": f"{score:.6f}",
                    "peak_gap": f"{peak_gap:.6f}",
                    "prelabel_method": "temporal_local_appearance",
                    "box_width_px": str(box[2] - box[0]),
                    "box_height_px": str(box[3] - box[1]),
                }
            )


def video_metadata(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {path}")
    metadata = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "reported_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return metadata


def read_yolo_box(path: Path, width: int, height: int) -> tuple[int, int, int, int] | None:
    fields = path.read_text(encoding="ascii").strip().split()
    if not fields:
        return None
    center_x, center_y, box_width, box_height = (float(value) for value in fields[1:])
    return (
        round((center_x - box_width / 2) * width),
        round((center_y - box_height / 2) * height),
        round((center_x + box_width / 2) * width),
        round((center_y + box_height / 2) * height),
    )


def make_contact_sheet(dataset: Path, rows: list[dict[str, str]]) -> Path:
    selected: list[dict[str, str]] = []
    for kind in ("normal", "fast", "empty"):
        group = [row for row in rows if row["source_kind"] == kind]
        for index in np.linspace(0, len(group) - 1, min(12, len(group)), dtype=int):
            selected.append(group[int(index)])
    tiles: list[np.ndarray] = []
    for row in selected:
        image_path = dataset / row["relative_image"]
        tile = cv2.imread(str(image_path))
        if tile is None:
            raise RuntimeError(f"unable to read generated image: {image_path}")
        label_path = dataset / "labels" / row["split"] / f"{image_path.stem}.txt"
        box = read_yolo_box(label_path, tile.shape[1], tile.shape[0])
        if box is not None:
            color = (0, 165, 255) if row["uncertain"] == "True" else (0, 220, 0)
            cv2.rectangle(tile, box[:2], box[2:], color, 2)
        caption = f"{row['source_kind']} f={row['source_frame']} {'uncertain' if row['uncertain'] == 'True' else ''}"
        cv2.putText(tile, caption, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(tile, caption, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(tile)
    rows_of_tiles = [np.hstack(tiles[index : index + 4]) for index in range(0, len(tiles), 4)]
    sheet = np.vstack(rows_of_tiles)
    output = dataset / "review" / "prelabel_contact_sheet.jpg"
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return output


def write_review_readme(dataset: Path, pending: int, empty: int) -> None:
    text = f"""# 视频适配数据人工复核

数据集：`{dataset}`

- 待复核有球帧：{pending}
- 已确认空轨道硬负样本：{empty}
- 所有候选框均由纯 OpenCV 生成，没有使用神经网络预标注。

启动命令：

```powershell
python {PROJECT_ROOT / 'scripts' / 'review_labels.py'} --dataset {dataset} --pending-only
```

按键：

- 框正确：按空格或 Enter 接受并前进。
- 框错误：按住鼠标左键重新拖框，再按空格或 Enter。
- 实际没有球：按 `N`，保存空标签并前进。
- `A`/左方向键：上一张；`D`/右方向键：下一张。
- `R`：恢复 OpenCV 原始候选框。
- `Q` 或 Esc：保存进度并退出，可稍后继续。

高速拖影应框住完整可见拖影，而不只是最暗的球心。到达 `EMPTY accepted` 图片时，有球帧已经复核完成，可以按 `Q` 退出。
"""
    (dataset / "README_REVIEW.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an isolated, manually reviewed video adaptation dataset.")
    parser.add_argument("--normal", type=Path, default=DEFAULT_NORMAL)
    parser.add_argument("--fast", type=Path, default=DEFAULT_FAST)
    parser.add_argument("--empty", type=Path, default=DEFAULT_EMPTY)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--normal-stride", type=int, default=8)
    parser.add_argument("--fast-stride", type=int, default=3)
    parser.add_argument("--empty-stride", type=int, default=5)
    parser.add_argument("--roi", default="170,190,1240,290")
    parser.add_argument("--output-size", default="470,110")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "pipeline.yaml")
    args = parser.parse_args()

    source_paths = (args.normal, args.fast, args.empty)
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.dataset.exists() and any(args.dataset.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty adaptation dataset: {args.dataset}")
    if min(args.normal_stride, args.fast_stride, args.empty_stride) <= 0:
        raise ValueError("all strides must be positive")

    roi_values = parse_tuple(args.roi, 4, "--roi")
    roi = dict(zip(("x", "y", "width", "height"), roi_values))
    output_size = parse_tuple(args.output_size, 2, "--output-size")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prelabel_config = {
        key: value for key, value in config["prelabel"].items() if key != "background_samples"
    }

    for relative in (
        "images/train",
        "labels/train",
        "labels_auto/train",
        "review",
        "calibration",
    ):
        (args.dataset / relative).mkdir(parents=True, exist_ok=True)

    specifications = (
        ("normal", args.normal, args.normal_stride, True),
        ("fast", args.fast, args.fast_stride, True),
        ("empty", args.empty, args.empty_stride, False),
    )
    metadata = {kind: video_metadata(path) for kind, path, _, _ in specifications}
    rows: list[dict[str, str]] = []
    decoded_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}

    for kind, video_path, stride, has_ball in specifications:
        detector = None
        background = None
        if has_ball:
            background = sampled_background(video_path, roi, output_size)
            cv2.imwrite(str(args.dataset / "calibration" / f"{kind}_median_background.png"), background)
            detector = TrackBallPrelabeler(background=background, **prelabel_config)

        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_index = 0
        selected = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride:
                frame_index += 1
                continue
            crop = crop_and_resize(frame, roi, output_size)
            height, width = crop.shape[:2]
            stem = f"{kind}_{frame_index:06d}"
            image_path = args.dataset / "images" / "train" / f"{stem}.jpg"
            label_path = args.dataset / "labels" / "train" / f"{stem}.txt"
            auto_path = args.dataset / "labels_auto" / "train" / f"{stem}.txt"
            if not cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"unable to write image: {image_path}")

            if detector is None:
                label_text = ""
                uncertain = False
                score = ""
                peak_gap = ""
                status = "accepted"
                expected = "0"
                method = "confirmed_empty_video"
                box_width = box_height = "0"
            else:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                detection = detector.locate(gray)
                box, method, unusual_component = motion_expanded_box(
                    gray,
                    background,
                    detection,
                    int(prelabel_config["search_y_min"]),
                    int(prelabel_config["search_y_max"]),
                )
                label_text = box_to_yolo(box, width, height)
                uncertain = bool(detection.uncertain or unusual_component)
                score = f"{detection.score:.6f}"
                peak_gap = f"{detection.peak_gap:.6f}"
                status = "pending"
                expected = "1"
                box_width = str(box[2] - box[0])
                box_height = str(box[3] - box[1])

            label_path.write_text(label_text, encoding="ascii")
            auto_path.write_text(label_text, encoding="ascii")
            rows.append(
                {
                    "relative_image": image_path.relative_to(args.dataset).as_posix(),
                    "split": "train",
                    "status": status,
                    "uncertain": str(uncertain),
                    "score": score,
                    "peak_gap": peak_gap,
                    "object_expected": expected,
                    "source_kind": kind,
                    "source_video": str(video_path.resolve()),
                    "source_frame": str(frame_index),
                    "source_time_ms": f"{frame_index * 1000.0 / fps:.3f}",
                    "prelabel_method": method,
                    "box_width_px": box_width,
                    "box_height_px": box_height,
                }
            )
            selected += 1
            frame_index += 1
        capture.release()
        decoded_counts[kind] = frame_index
        selected_counts[kind] = selected
        print(f"{kind}: decoded={frame_index}, selected={selected}, stride={stride}")

    refine_sequence_labels(args.dataset, rows)

    status_path = args.dataset / "review" / "status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    pending_count = sum(row["status"] == "pending" for row in rows)
    empty_count = sum(row["object_expected"] == "0" for row in rows)
    contact_sheet = make_contact_sheet(args.dataset, rows)
    write_review_readme(args.dataset, pending_count, empty_count)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "diagnostic videos repurposed as adaptation data; not valid for final acceptance",
        "dataset_root": str(args.dataset.resolve()),
        "source_videos_read_only": True,
        "prelabel": "pure OpenCV local darkness, neighboring-frame change, and median-background fallback",
        "roi": roi,
        "output_size": list(output_size),
        "strides": {kind: stride for kind, _, stride, _ in specifications},
        "videos": metadata,
        "decoded_frames": decoded_counts,
        "selected_frames": selected_counts,
        "total_images": len(rows),
        "pending_positive_images": pending_count,
        "accepted_empty_images": empty_count,
        "contact_sheet": str(contact_sheet.resolve()),
    }
    (args.dataset / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Review with: python {PROJECT_ROOT / 'scripts' / 'review_labels.py'} --dataset {args.dataset} --pending-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
