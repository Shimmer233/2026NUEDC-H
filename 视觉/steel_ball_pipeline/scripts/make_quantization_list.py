from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.dataset import label_for_image, verify_approval  # noqa: E402


def bin_index(values: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, bins - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a stratified train-only RKNN calibration list.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\University\NUEDC\datasets\steel_ball_v2"),
    )
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts" / "quantization.txt"
    )
    args = parser.parse_args()
    approved, reason = verify_approval(args.dataset)
    if not approved:
        raise RuntimeError(f"quantization list blocked: {reason}")
    paths = sorted((args.dataset / "images" / "train").glob("*.jpg"))
    if len(paths) < args.count:
        raise RuntimeError(f"requested {args.count} calibration images, only {len(paths)} exist")
    metrics: list[tuple[Path, float, float, float]] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"unable to read {path}")
        fields = label_for_image(args.dataset, path).read_text(encoding="ascii").split()
        x_center = float(fields[1]) if fields else -1.0
        brightness = float(image.mean())
        sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
        metrics.append((path, x_center, brightness, sharpness))
    values = np.asarray([[item[1], item[2], item[3]] for item in metrics])
    position_bins = np.full(len(values), -1, dtype=np.int64)
    positive = values[:, 0] >= 0
    position_bins[positive] = bin_index(values[positive, 0], 6)
    brightness_bins = bin_index(values[:, 1], 4)
    sharpness_bins = bin_index(values[:, 2], 3)
    strata: dict[tuple[int, int, int], deque[Path]] = defaultdict(deque)
    rng = np.random.default_rng(42)
    for key in sorted(set(zip(position_bins, brightness_bins, sharpness_bins))):
        indices = np.flatnonzero(
            (position_bins == key[0])
            & (brightness_bins == key[1])
            & (sharpness_bins == key[2])
        )
        rng.shuffle(indices)
        strata[key].extend(metrics[int(index)][0] for index in indices)
    selected: list[Path] = []
    while len(selected) < args.count:
        progressed = False
        for key in sorted(strata):
            if strata[key] and len(selected) < args.count:
                selected.append(strata[key].popleft())
                progressed = True
        if not progressed:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(path.resolve().as_posix() for path in selected) + "\n"
    args.output.write_text(text, encoding="utf-8")
    negative_selected = sum(
        not label_for_image(args.dataset, path).read_text(encoding="ascii").strip()
        for path in selected
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "requested": args.count,
        "selected": len(selected),
        "source_split": "train",
        "uses_validation_or_test": False,
        "negative_images_available": int(np.sum(~positive)),
        "negative_images_selected": negative_selected,
        "selection": "round-robin across 6 position x 4 brightness x 3 sharpness quantile bins",
        "list_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
