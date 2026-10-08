from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.import_test_video import crop_and_resize, sampled_background  # noqa: E402
from scripts.prepare_dataset import capture_key  # noqa: E402
from scripts.prepare_video_adaptation import (  # noqa: E402
    box_to_yolo,
    motion_expanded_box,
    refine_sequence_labels,
    video_metadata,
)
from steel_ball.dataset import (  # noqa: E402
    image_files,
    label_for_image,
    read_review_rows,
    verify_approval,
)
from steel_ball.prelabel import Detection, TrackBallPrelabeler, yolo_line  # noqa: E402


DEFAULT_BASE = Path(r"D:\University\NUEDC\datasets\steel_ball_v3_adapted")
DEFAULT_MEDIA = Path(r"C:\Users\Light\Pictures\Camera Roll")
DEFAULT_OUTPUT = Path(r"D:\University\NUEDC\datasets\steel_ball_v4_incremental")
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "incremental_v4.yaml"
STATUS_FIELDS = [
    "relative_image",
    "split",
    "status",
    "uncertain",
    "score",
    "peak_gap",
    "object_expected",
    "source_dataset",
    "source_kind",
    "source_path",
    "source_frame",
    "source_time_ms",
    "prelabel_method",
    "box_width_px",
    "box_height_px",
    "review_action",
]


@dataclass(frozen=True)
class PhotoProposal:
    path: Path
    timestamp: datetime
    sequence: int
    detection: Detection
    sharpness: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_tuple(value: str, count: int, name: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if len(result) != count:
        raise argparse.ArgumentTypeError(
            f"{name} requires {count} comma-separated integers"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a v4 dataset from approved v3 data and newly captured media."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--media", type=Path, default=DEFAULT_MEDIA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty dataset: {path}")
    for split in ("train", "val", "test"):
        for root in ("images", "labels", "labels_auto"):
            (path / root / split).mkdir(parents=True, exist_ok=True)
    for relative in ("review", "manifests", "calibration"):
        (path / relative).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(dataset: Path) -> None:
    payload = {
        "path": dataset.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["ball"],
    }
    (dataset / "steel_ball.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def base_status_row(
    base: Path,
    destination: Path,
    source_image: Path,
    source_row: dict[str, str],
) -> dict[str, str]:
    relative = destination.relative_to(destination.parents[2]).as_posix()
    return {
        "relative_image": relative,
        "split": destination.parent.name,
        "status": "accepted",
        "uncertain": source_row.get("uncertain", "False"),
        "score": source_row.get("score", ""),
        "peak_gap": source_row.get("peak_gap", ""),
        "object_expected": source_row.get("object_expected", "1"),
        "source_dataset": "base_v3",
        "source_kind": source_row.get("source_kind", "base_v3"),
        "source_path": str(source_image.resolve()),
        "source_frame": source_row.get("source_frame", ""),
        "source_time_ms": source_row.get("source_time_ms", ""),
        "prelabel_method": "inherited_reviewed_label",
        "box_width_px": source_row.get("box_width_px", ""),
        "box_height_px": source_row.get("box_height_px", ""),
        "review_action": "inherited",
    }


def copy_approved_base(base: Path, output: Path) -> tuple[list[dict[str, str]], list[dict]]:
    approved, reason = verify_approval(base)
    if not approved:
        raise RuntimeError(f"base dataset is not approved: {reason}")
    source_rows = read_review_rows(base)
    rows: list[dict[str, str]] = []
    inventory: list[dict] = []
    for source_image in image_files(base, ("train", "val")):
        source_relative = source_image.relative_to(base).as_posix()
        source_row = source_rows.get(source_relative)
        if source_row is None or source_row.get("status") != "accepted":
            raise RuntimeError(f"approved base row is missing: {source_relative}")
        split = source_image.parent.name
        destination_image = output / "images" / split / source_image.name
        destination_label = output / "labels" / split / f"{source_image.stem}.txt"
        destination_auto = output / "labels_auto" / split / f"{source_image.stem}.txt"
        source_label = label_for_image(base, source_image)
        link_or_copy(source_image, destination_image)
        link_or_copy(source_label, destination_label)
        link_or_copy(source_label, destination_auto)
        rows.append(base_status_row(base, destination_image, source_image, source_row))
        inventory.append(
            {
                "relative_image": destination_image.relative_to(output).as_posix(),
                "source_dataset": "base_v3",
                "source_path": str(source_image.resolve()),
                "image_sha256": sha256(source_image),
                "label_sha256": sha256(source_label),
            }
        )
    return rows, inventory


def read_photo_crop(path: Path, roi: dict, output_size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"unable to read image: {path}")
    return crop_and_resize(image, roi, output_size)


def photo_background(
    paths: list[Path], roi: dict, output_size: tuple[int, int], sample_count: int
) -> np.ndarray:
    indices = np.linspace(0, len(paths) - 1, min(sample_count, len(paths)), dtype=int)
    samples = [
        cv2.cvtColor(read_photo_crop(paths[int(index)], roi, output_size), cv2.COLOR_BGR2GRAY)
        for index in indices
    ]
    return np.median(np.stack(samples), axis=0).astype(np.uint8)


def analyze_photos(
    paths: list[Path],
    roi: dict,
    output_size: tuple[int, int],
    detector: TrackBallPrelabeler,
) -> list[PhotoProposal]:
    proposals: list[PhotoProposal] = []
    for index, path in enumerate(paths, start=1):
        crop = read_photo_crop(path, roi, output_size)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        timestamp, sequence, _ = capture_key(path)
        proposals.append(
            PhotoProposal(
                path=path,
                timestamp=timestamp,
                sequence=sequence,
                detection=detector.locate(gray),
                sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            )
        )
        if index % 100 == 0 or index == len(paths):
            print(f"analyzed photos {index}/{len(paths)}")
    return proposals


def proposal_time(proposal: PhotoProposal) -> float:
    return proposal.timestamp.timestamp() + proposal.sequence / 1000.0


def select_photos(
    proposals: list[PhotoProposal],
    minimum_motion_px: float,
    maximum_interval_seconds: float,
    endpoint_margin_px: float,
    blur_quantile: float,
) -> list[PhotoProposal]:
    if not proposals:
        return []
    sharpness_cutoff = float(
        np.quantile([proposal.sharpness for proposal in proposals], blur_quantile)
    )
    selected: list[PhotoProposal] = []
    last: PhotoProposal | None = None
    for proposal in proposals:
        detection = proposal.detection
        distance = (
            float("inf")
            if last is None
            else float(
                np.hypot(
                    detection.center_x - last.detection.center_x,
                    detection.center_y - last.detection.center_y,
                )
            )
        )
        elapsed = (
            float("inf")
            if last is None
            else proposal_time(proposal) - proposal_time(last)
        )
        endpoint = (
            detection.center_x <= endpoint_margin_px
            or detection.center_x >= 470 - endpoint_margin_px
        )
        keep = (
            last is None
            or distance >= minimum_motion_px
            or elapsed >= maximum_interval_seconds
            or endpoint
            or detection.uncertain
            or proposal.sharpness <= sharpness_cutoff
        )
        if keep:
            selected.append(proposal)
            last = proposal
    return selected


def write_prelabel(
    output: Path,
    split: str,
    stem: str,
    crop: np.ndarray,
    label_text: str,
) -> tuple[Path, Path]:
    image_path = output / "images" / split / f"{stem}.jpg"
    label_path = output / "labels" / split / f"{stem}.txt"
    auto_path = output / "labels_auto" / split / f"{stem}.txt"
    if not cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"unable to write image: {image_path}")
    label_path.write_text(label_text, encoding="ascii")
    auto_path.write_text(label_text, encoding="ascii")
    return image_path, label_path


def add_photos(
    proposals: list[PhotoProposal],
    output: Path,
    roi: dict,
    output_size: tuple[int, int],
) -> tuple[list[dict[str, str]], list[dict]]:
    rows: list[dict[str, str]] = []
    inventory: list[dict] = []
    for index, proposal in enumerate(proposals, start=1):
        crop = read_photo_crop(proposal.path, roi, output_size)
        stem = f"newphoto__{proposal.path.stem}"
        label_text = yolo_line(proposal.detection, crop.shape[1], crop.shape[0])
        image_path, label_path = write_prelabel(output, "train", stem, crop, label_text)
        detection = proposal.detection
        rows.append(
            {
                "relative_image": image_path.relative_to(output).as_posix(),
                "split": "train",
                "status": "pending",
                "uncertain": str(detection.uncertain),
                "score": f"{detection.score:.6f}",
                "peak_gap": f"{detection.peak_gap:.6f}",
                "object_expected": "1",
                "source_dataset": "new_media",
                "source_kind": "photo",
                "source_path": str(proposal.path.resolve()),
                "source_frame": "",
                "source_time_ms": "",
                "prelabel_method": "opencv_background_contrast",
                "box_width_px": "",
                "box_height_px": "",
                "review_action": "",
            }
        )
        inventory.append(
            {
                "relative_image": image_path.relative_to(output).as_posix(),
                "source_dataset": "new_media",
                "source_path": str(proposal.path.resolve()),
                "source_sha256": sha256(proposal.path),
                "image_sha256": sha256(image_path),
                "label_sha256": sha256(label_path),
                "sharpness": proposal.sharpness,
            }
        )
        if index % 100 == 0 or index == len(proposals):
            print(f"wrote selected photos {index}/{len(proposals)}")
    return rows, inventory


def add_video(
    output: Path,
    kind: str,
    video_path: Path,
    stride: int,
    has_ball: bool,
    train_fraction: float,
    roi: dict,
    output_size: tuple[int, int],
    prelabel_config: dict,
) -> tuple[list[dict[str, str]], list[dict]]:
    metadata = video_metadata(video_path)
    frame_count = int(metadata["reported_frames"])
    split_frame = int(frame_count * train_fraction)
    fps = float(metadata["fps"])
    detector = None
    background = None
    if has_ball:
        background = sampled_background(video_path, roi, output_size)
        cv2.imwrite(str(output / "calibration" / f"{kind}_background.png"), background)
        detector = TrackBallPrelabeler(background=background, **prelabel_config)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    rows: list[dict[str, str]] = []
    inventory: list[dict] = []
    frame_index = 0
    selected = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride:
            frame_index += 1
            continue
        split = "train" if frame_index < split_frame else "val"
        crop = crop_and_resize(frame, roi, output_size)
        height, width = crop.shape[:2]
        stem = f"newvideo__{kind}_{frame_index:06d}"
        if detector is None:
            label_text = ""
            uncertain = False
            score = peak_gap = ""
            expected = "0"
            method = "empty_video_candidate"
            box_width = box_height = "0"
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            detection = detector.locate(gray)
            box, method, unusual = motion_expanded_box(
                gray,
                background,
                detection,
                int(prelabel_config["search_y_min"]),
                int(prelabel_config["search_y_max"]),
            )
            label_text = box_to_yolo(box, width, height)
            uncertain = bool(detection.uncertain or unusual)
            score = f"{detection.score:.6f}"
            peak_gap = f"{detection.peak_gap:.6f}"
            expected = "1"
            box_width = str(box[2] - box[0])
            box_height = str(box[3] - box[1])
        image_path, label_path = write_prelabel(output, split, stem, crop, label_text)
        rows.append(
            {
                "relative_image": image_path.relative_to(output).as_posix(),
                "split": split,
                "status": "pending",
                "uncertain": str(uncertain),
                "score": score,
                "peak_gap": peak_gap,
                "object_expected": expected,
                "source_dataset": "new_media",
                "source_kind": kind,
                "source_path": str(video_path.resolve()),
                "source_frame": str(frame_index),
                "source_time_ms": f"{frame_index * 1000.0 / fps:.3f}",
                "prelabel_method": method,
                "box_width_px": box_width,
                "box_height_px": box_height,
                "review_action": "",
            }
        )
        inventory.append(
            {
                "relative_image": image_path.relative_to(output).as_posix(),
                "source_dataset": "new_media",
                "source_path": str(video_path.resolve()),
                "source_frame": frame_index,
                "source_video_sha256": metadata["sha256"],
                "image_sha256": sha256(image_path),
                "label_sha256": sha256(label_path),
            }
        )
        selected += 1
        frame_index += 1
    capture.release()
    print(f"{kind}: decoded={frame_index}, selected={selected}, stride={stride}")
    return rows, inventory


def normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{field: row.get(field, "") for field in STATUS_FIELDS} for row in rows]


def write_status(output: Path, rows: list[dict[str, str]]) -> None:
    with (output / "review" / "status.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(normalize_rows(rows))


def refresh_inventory_hashes(output: Path, inventory: list[dict]) -> None:
    for record in inventory:
        image_path = output / str(record["relative_image"])
        label_path = label_for_image(output, image_path)
        record["image_sha256"] = sha256(image_path)
        record["label_sha256"] = sha256(label_path)


def write_review_readme(output: Path, new_count: int) -> None:
    text = f"""# Incremental v4 label review

Dataset: `{output}`

All {new_count} newly generated images are pending and must be reviewed individually.
Previously approved v3 rows are inherited and do not need another review.

Start the editable, mandatory review (recommended):

```powershell
{PROJECT_ROOT / 'review_v4_incremental.cmd'}
```

The full track remains visible. Drag with the left mouse button to replace the box, then
press Enter or Space to accept it. Press N for an empty image.

Optional read-only inspection (cannot redraw or accept):

```powershell
{PROJECT_ROOT / 'view_v4_prelabels_readonly.cmd'}
```

- Drag with the left mouse button to replace the box.
- Enter or Space accepts the current box.
- N confirms that the image contains no ball.
- R restores the original automatic label.
- A/D or Left/Right moves without accepting an image.
- Q or Escape saves progress and exits.

Training stays blocked until every pending image is explicitly accepted.
"""
    (output / "README_REVIEW.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    approved, reason = verify_approval(args.base)
    if not approved:
        raise RuntimeError(f"base dataset is not approved: {reason}")
    if not args.media.is_dir():
        raise FileNotFoundError(args.media)
    photos = sorted(args.media.glob("*.jpg"), key=capture_key)
    if not photos:
        raise RuntimeError(f"no JPG images found in {args.media}")

    videos = {
        "normal": args.media / str(config["videos"]["normal"]),
        "fast": args.media / str(config["videos"]["fast"]),
        "empty": args.media / str(config["videos"]["empty"]),
    }
    for video in videos.values():
        if not video.is_file():
            raise FileNotFoundError(video)

    prepare_output(args.output)
    photo_roi = dict(zip(("x", "y", "width", "height"), parse_tuple(
        str(config["photo_roi"]), 4, "photo_roi"
    )))
    video_roi = dict(zip(("x", "y", "width", "height"), parse_tuple(
        str(config["video_roi"]), 4, "video_roi"
    )))
    output_size = parse_tuple(str(config["output_size"]), 2, "output_size")
    prelabel_config = dict(config["prelabel"])

    rows, inventory = copy_approved_base(args.base, args.output)
    base_count = len(rows)

    background = photo_background(
        photos,
        photo_roi,
        output_size,
        int(config["photo_selection"]["background_samples"]),
    )
    cv2.imwrite(str(args.output / "calibration" / "photo_background.png"), background)
    photo_detector = TrackBallPrelabeler(background=background, **prelabel_config)
    proposals = analyze_photos(photos, photo_roi, output_size, photo_detector)
    selected_photos = select_photos(
        proposals,
        float(config["photo_selection"]["minimum_motion_px"]),
        float(config["photo_selection"]["maximum_interval_seconds"]),
        float(config["photo_selection"]["endpoint_margin_px"]),
        float(config["photo_selection"]["blur_quantile"]),
    )
    photo_rows, photo_inventory = add_photos(
        selected_photos, args.output, photo_roi, output_size
    )
    rows.extend(photo_rows)
    inventory.extend(photo_inventory)

    all_video_rows: list[dict[str, str]] = []
    for kind, has_ball in (("normal", True), ("fast", True), ("empty", False)):
        video_rows, video_inventory = add_video(
            args.output,
            kind,
            videos[kind],
            int(config["video_strides"][kind]),
            has_ball,
            float(config["train_fraction"]),
            video_roi,
            output_size,
            prelabel_config,
        )
        all_video_rows.extend(video_rows)
        inventory.extend(video_inventory)

    refine_sequence_labels(args.output, all_video_rows)
    rows.extend(all_video_rows)
    refresh_inventory_hashes(args.output, inventory)
    write_status(args.output, rows)
    write_dataset_yaml(args.output)
    new_count = len(rows) - base_count
    write_review_readme(args.output, new_count)

    inventory_text = json.dumps(inventory, sort_keys=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.output.resolve()),
        "base_dataset": str(args.base.resolve()),
        "base_approval": json.loads(
            (args.base / ".review-approved.json").read_text(encoding="utf-8")
        ),
        "source_media": str(args.media.resolve()),
        "source_media_mutated": False,
        "photo_source_count": len(photos),
        "photo_selected_count": len(selected_photos),
        "video_metadata": {kind: video_metadata(path) for kind, path in videos.items()},
        "base_image_count": base_count,
        "new_pending_count": new_count,
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "val", "test")
        },
        "photo_roi": photo_roi,
        "video_roi": video_roi,
        "output_size": list(output_size),
        "config": str(args.config.resolve()),
        "inventory_sha256": hashlib.sha256(inventory_text.encode("utf-8")).hexdigest(),
        "inventory": inventory,
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
    for name in ("build.json", "source_inventory.json"):
        (args.output / "manifests" / name).write_text(manifest_text, encoding="utf-8")
    summary = {key: value for key, value in manifest.items() if key != "inventory"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        "Review all new labels with: "
        f"python {PROJECT_ROOT / 'scripts' / 'review_labels.py'} "
        f"--dataset {args.output} --pending-only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
