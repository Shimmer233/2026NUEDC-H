from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.dataset import image_files, label_for_image, validate_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate labels and render prelabel contact sheets.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\University\NUEDC\datasets\steel_ball_v2"),
    )
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def load_status(dataset: Path) -> dict[str, dict[str, str]]:
    with (dataset / "review" / "status.csv").open("r", newline="", encoding="utf-8") as stream:
        return {row["relative_image"]: row for row in csv.DictReader(stream)}


def pixel_box(label_path: Path, width: int, height: int) -> tuple[int, int, int, int] | None:
    fields = label_path.read_text(encoding="ascii").strip().split()
    if not fields:
        return None
    center_x, center_y, box_width, box_height = (float(value) for value in fields[1:])
    return (
        round((center_x - box_width / 2) * width),
        round((center_y - box_height / 2) * height),
        round((center_x + box_width / 2) * width),
        round((center_y + box_height / 2) * height),
    )


def render_sheets(dataset: Path, images: list[Path], statuses: dict[str, dict[str, str]]) -> int:
    output_dir = dataset / "review" / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    columns, rows_per_page, scale = 2, 4, 2
    tile_width, image_height, header_height = 470 * scale, 110 * scale, 24
    tile_height = image_height + header_height
    per_page = columns * rows_per_page
    page_count = math.ceil(len(images) / per_page)
    for page_index in range(page_count):
        canvas = np.full((tile_height * rows_per_page, tile_width * columns, 3), 42, dtype=np.uint8)
        page_images = images[page_index * per_page : (page_index + 1) * per_page]
        for tile_index, image_path in enumerate(page_images):
            relative = image_path.relative_to(dataset).as_posix()
            status = statuses[relative]
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            box = pixel_box(label_for_image(dataset, image_path), image.shape[1], image.shape[0])
            if status["status"] == "accepted":
                color = (0, 210, 0)
            elif status["uncertain"].lower() == "true":
                color = (0, 0, 255)
            else:
                color = (0, 190, 255)
            if box is not None:
                x0, y0, x1, y1 = box
                cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
            enlarged = cv2.resize(image, (tile_width, image_height), interpolation=cv2.INTER_NEAREST)
            row, column = divmod(tile_index, columns)
            origin_x, origin_y = column * tile_width, row * tile_height
            canvas[origin_y + header_height : origin_y + tile_height, origin_x : origin_x + tile_width] = enlarged
            text = (
                f"{page_index * per_page + tile_index + 1:04d} "
                f"{status['status']} U={status['uncertain']} S={float(status['score']):.2f} "
                f"{image_path.name}"
            )
            cv2.putText(
                canvas,
                text,
                (origin_x + 4, origin_y + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(
            str(output_dir / f"page_{page_index + 1:03d}.jpg"),
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
    return page_count


def main() -> int:
    args = parse_args()
    images = image_files(args.dataset, ("train", "val", "test"))
    statuses = load_status(args.dataset)
    errors: list[str] = []
    widths: list[float] = []
    heights: list[float] = []
    split_counts: dict[str, int] = {}
    for image_path in images:
        relative = image_path.relative_to(args.dataset).as_posix()
        split_counts[image_path.parent.name] = split_counts.get(image_path.parent.name, 0) + 1
        if relative not in statuses:
            errors.append(f"missing status: {relative}")
            continue
        label_path = label_for_image(args.dataset, image_path)
        require_one = statuses[relative].get("object_expected", "1") == "1"
        errors.extend(
            f"{relative}: {item}" for item in validate_label(label_path, require_one=require_one)
        )
        fields = label_path.read_text(encoding="ascii").strip().split()
        if len(fields) == 5:
            widths.append(float(fields[3]) * 470)
            heights.append(float(fields[4]) * 110)
    uncertain = sum(row["uncertain"].lower() == "true" for row in statuses.values())
    accepted = sum(row["status"] == "accepted" for row in statuses.values())
    if args.no_render:
        page_count = len(list((args.dataset / "review" / "contact_sheets").glob("page_*.jpg")))
    else:
        page_count = 0 if errors else render_sheets(args.dataset, images, statuses)
    report = {
        "image_count": len(images),
        "split_counts": split_counts,
        "status_count": len(statuses),
        "accepted": accepted,
        "uncertain": uncertain,
        "label_errors": len(errors),
        "box_width_px": {
            "min": min(widths, default=0),
            "median": float(np.median(widths)) if widths else 0,
            "max": max(widths, default=0),
        },
        "box_height_px": {
            "min": min(heights, default=0),
            "median": float(np.median(heights)) if heights else 0,
            "max": max(heights, default=0),
        },
        "contact_sheet_pages": page_count,
        "errors": errors[:100],
    }
    report_path = args.dataset / "review" / "audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
