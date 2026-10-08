from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.review_labels import read_box, read_status  # noqa: E402


WINDOW_NAME = "Steel ball prelabel inspection (read only)"
DEFAULT_DATASET = Path(r"D:\University\NUEDC\datasets\steel_ball_v4_incremental")
HEADER_HEIGHT = 64
DETAIL_WIDTH = 240
PADDING = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect automatic steel-ball boxes without changing review status or labels."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--include-old",
        action="store_true",
        help="Include inherited images as well as newly generated images.",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Only show rows whose review status is still pending.",
    )
    parser.add_argument(
        "--uncertain-only",
        action="store_true",
        help="Only show proposals marked uncertain by the automatic prelabeler.",
    )
    parser.add_argument("--scale", type=int, choices=(1, 2, 3), default=2)
    return parser.parse_args()


def select_rows(
    rows: list[dict[str, str]],
    *,
    include_old: bool,
    pending_only: bool,
    uncertain_only: bool,
) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        if not include_old and row.get("source_dataset") != "new_media":
            continue
        if pending_only and row.get("status") == "accepted":
            continue
        if uncertain_only and row.get("uncertain", "").lower() != "true":
            continue
        selected.append(row)
    return selected


def auto_label_path(dataset: Path, row: dict[str, str]) -> Path:
    image_path = dataset / row["relative_image"]
    return dataset / "labels_auto" / row["split"] / f"{image_path.stem}.txt"


def _put_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (235, 235, 235),
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)


def _detail_view(
    image: np.ndarray,
    box: tuple[int, int, int, int] | None,
    height: int,
) -> np.ndarray:
    panel = np.full((height, DETAIL_WIDTH, 3), 28, dtype=np.uint8)
    if box is None:
        _put_text(panel, "EMPTY PRELABEL", (45, max(28, height // 2)))
        return panel

    image_height, image_width = image.shape[:2]
    x0, y0, x1, y1 = box
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    radius = max(18, int(max(x1 - x0, y1 - y0) * 2.5))
    crop_x0 = max(0, center_x - radius)
    crop_y0 = max(0, center_y - radius)
    crop_x1 = min(image_width, center_x + radius)
    crop_y1 = min(image_height, center_y + radius)
    crop = image[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    if not crop.size:
        return panel

    cv2.rectangle(
        crop,
        (x0 - crop_x0, y0 - crop_y0),
        (x1 - crop_x0, y1 - crop_y0),
        (0, 255, 0),
        1,
    )
    available_height = max(1, height - 2 * PADDING)
    factor = min(
        (DETAIL_WIDTH - 2 * PADDING) / crop.shape[1],
        available_height / crop.shape[0],
    )
    resized = cv2.resize(
        crop,
        None,
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_NEAREST,
    )
    offset_x = (DETAIL_WIDTH - resized.shape[1]) // 2
    offset_y = (height - resized.shape[0]) // 2
    panel[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
    return panel


def render_item(
    dataset: Path,
    row: dict[str, str],
    index: int,
    total: int,
    scale: int,
) -> np.ndarray:
    image_path = dataset / row["relative_image"]
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"unable to read image: {image_path}")
    label_path = auto_label_path(dataset, row)
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    height, width = image.shape[:2]
    box = read_box(label_path, width, height)

    full_track = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    if box is not None:
        x0, y0, x1, y1 = box
        color = (0, 190, 255) if row.get("uncertain", "").lower() == "true" else (0, 255, 0)
        cv2.rectangle(
            full_track,
            (x0 * scale, y0 * scale),
            (x1 * scale, y1 * scale),
            color,
            max(2, scale),
        )
    cv2.rectangle(
        full_track,
        (0, 0),
        (full_track.shape[1] - 1, full_track.shape[0] - 1),
        (255, 255, 255),
        1,
    )

    detail = _detail_view(image, box, full_track.shape[0])
    content = np.hstack((full_track, detail))
    canvas = cv2.copyMakeBorder(
        content,
        HEADER_HEIGHT,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(28, 28, 28),
    )
    state = "BALL" if box is not None else "EMPTY"
    uncertainty = "UNCERTAIN" if row.get("uncertain", "").lower() == "true" else "normal"
    _put_text(
        canvas,
        f"{index + 1}/{total}  AUTO {state}  {uncertainty}  review={row.get('status', '')}",
        (PADDING, 23),
        (80, 230, 255) if uncertainty == "UNCERTAIN" else (235, 235, 235),
    )
    _put_text(canvas, Path(row["relative_image"]).name, (PADDING, 49))
    return canvas


def main() -> int:
    args = parse_args()
    status_path = args.dataset / "review" / "status.csv"
    rows, _ = read_status(status_path)
    rows = select_rows(
        rows,
        include_old=args.include_old,
        pending_only=args.pending_only,
        uncertain_only=args.uncertain_only,
    )
    if not rows:
        print("No prelabels match the selected filters.")
        return 0

    print("Read-only viewer: A/Left previous, D/Right next, Home/End jump, Q/Esc quit.")
    print("The complete training ROI stays visible on the left; the ball detail is on the right.")
    index = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    while True:
        cv2.imshow(WINDOW_NAME, render_item(args.dataset, rows[index], index, len(rows), args.scale))
        key = cv2.waitKeyEx(0)
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("d"), ord("D"), 2555904):
            index = min(len(rows) - 1, index + 1)
        elif key in (ord("a"), ord("A"), 2424832):
            index = max(0, index - 1)
        elif key in (2359296,):
            index = 0
        elif key in (2293760,):
            index = len(rows) - 1
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
