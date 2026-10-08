from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from steel_ball.dataset import label_for_image, verify_approval


def _eligible_for_synthetic_blur(image_path: Path) -> bool:
    """Real video adaptation frames already contain authentic motion blur."""
    return not image_path.stem.startswith("adapt__")


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _motion_kernel(size: int, angle_degrees: float) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2
    radians = np.deg2rad(angle_degrees)
    delta_x = np.cos(radians) * center
    delta_y = np.sin(radians) * center
    first = (round(center - delta_x), round(center - delta_y))
    second = (round(center + delta_x), round(center + delta_y))
    cv2.line(kernel, first, second, 1.0, 1)
    total = float(kernel.sum())
    return kernel / total if total else np.eye(size, dtype=np.float32) / size


def _expanded_motion_label(
    label_text: str,
    image_width: int,
    image_height: int,
    kernel_size: int,
    angle_degrees: float,
) -> str:
    fields = label_text.split()
    if not fields:
        return ""
    class_id, center_x, center_y, width, height = fields
    radians = np.deg2rad(angle_degrees)
    blur_length = kernel_size - 1
    expanded_width = min(1.0, float(width) + abs(np.cos(radians)) * blur_length / image_width)
    expanded_height = min(1.0, float(height) + abs(np.sin(radians)) * blur_length / image_height)
    return (
        f"{int(class_id)} {float(center_x):.8f} {float(center_y):.8f} "
        f"{expanded_width:.8f} {expanded_height:.8f}\n"
    )


def build_training_view(dataset_root: Path, seed: int = 42, fraction: float = 0.70) -> Path:
    approved, reason = verify_approval(dataset_root)
    if not approved:
        raise RuntimeError(f"training view blocked: {reason}")
    train_images = sorted((dataset_root / "images" / "train").glob("*.jpg"))
    view_root = dataset_root / "generated" / "training_view"
    temporary = dataset_root / "generated" / "training_view.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    image_dir = temporary / "images" / "train"
    label_dir = temporary / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for image_path in train_images:
        _link_or_copy(image_path, image_dir / image_path.name)
        _link_or_copy(label_for_image(dataset_root, image_path), label_dir / f"{image_path.stem}.txt")

    rng = np.random.default_rng(seed)
    blur_candidates = [image for image in train_images if _eligible_for_synthetic_blur(image)]
    augmented_count = round(len(blur_candidates) * fraction)
    selected_indices = rng.choice(len(blur_candidates), size=augmented_count, replace=False)
    records: list[dict[str, object]] = []
    for index in sorted(selected_indices.tolist()):
        source = blur_candidates[index]
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"unable to read {source}")
        kernel_size = int(rng.choice([3, 5, 7, 11, 15, 21, 31]))
        angle = float(rng.uniform(-6.0, 6.0))
        augmented = cv2.filter2D(image, -1, _motion_kernel(kernel_size, angle))
        stem = f"{source.stem}_motionblur"
        destination = image_dir / f"{stem}.jpg"
        if not cv2.imwrite(str(destination), augmented, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"unable to write {destination}")
        source_label = label_for_image(dataset_root, source).read_text(encoding="ascii")
        (label_dir / f"{stem}.txt").write_text(
            _expanded_motion_label(
                source_label,
                image.shape[1],
                image.shape[0],
                kernel_size,
                angle,
            ),
            encoding="ascii",
        )
        records.append(
            {
                "source": source.name,
                "generated": destination.name,
                "kernel_size": kernel_size,
                "angle_degrees": angle,
            }
        )
    manifest_text = json.dumps(records, sort_keys=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "original_count": len(train_images),
        "synthetic_blur_candidate_count": len(blur_candidates),
        "synthetic_blur_exclusion": "adapt__* real-video frames",
        "motion_blur_count": augmented_count,
        "total_count": len(train_images) + augmented_count,
        "motion_blur_fraction": fraction,
        "records_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "records": records,
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if view_root.exists():
        shutil.rmtree(view_root)
    temporary.replace(view_root)
    data_yaml = dataset_root / "generated" / "steel_ball_training.yaml"
    payload = {
        "path": dataset_root.as_posix(),
        "train": "generated/training_view/images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["ball"],
    }
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return data_yaml
