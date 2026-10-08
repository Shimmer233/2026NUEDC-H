from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.dataset import (  # noqa: E402
    APPROVAL_FILE,
    image_files,
    label_for_image,
    read_review_rows,
    verify_approval,
)


DEFAULT_BASE = Path(r"D:\University\NUEDC\datasets\steel_ball_v2")
DEFAULT_ADAPTATION = Path(r"D:\University\NUEDC\datasets\steel_ball_video_adaptation_v2")
DEFAULT_OUTPUT = Path(r"D:\University\NUEDC\datasets\steel_ball_v3_adapted")
STATUS_FIELDS = [
    "relative_image",
    "split",
    "status",
    "uncertain",
    "score",
    "peak_gap",
    "object_expected",
    "source_dataset",
    "source_relative_image",
    "source_kind",
    "source_frame",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_approval(dataset: Path) -> dict[str, object]:
    approved, reason = verify_approval(dataset)
    if not approved:
        raise RuntimeError(f"source dataset is not approved: {dataset}: {reason}")
    return json.loads((dataset / APPROVAL_FILE).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine reviewed base and video-adaptation data.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adaptation", type=Path, default=DEFAULT_ADAPTATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base_approval = require_approval(args.base)
    adaptation_approval = require_approval(args.adaptation)
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output dataset: {args.output}")
    for relative in (
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
        "review",
        "manifests",
    ):
        (args.output / relative).mkdir(parents=True, exist_ok=True)

    source_rows = {
        "base": read_review_rows(args.base),
        "adapt": read_review_rows(args.adaptation),
    }
    specifications = (
        ("base", args.base, ("train", "val")),
        ("adapt", args.adaptation, ("train",)),
    )
    combined_rows: list[dict[str, str]] = []
    inventory: list[dict[str, object]] = []
    names: set[str] = set()

    for source_name, source_root, splits in specifications:
        for image_path in image_files(source_root, splits):
            split = image_path.parent.name
            destination_name = f"{source_name}__{image_path.name}"
            if destination_name in names:
                raise RuntimeError(f"combined filename collision: {destination_name}")
            names.add(destination_name)
            source_label = label_for_image(source_root, image_path)
            destination_image = args.output / "images" / split / destination_name
            destination_label = args.output / "labels" / split / f"{Path(destination_name).stem}.txt"
            shutil.copy2(image_path, destination_image)
            shutil.copy2(source_label, destination_label)

            source_relative = image_path.relative_to(source_root).as_posix()
            source_status = source_rows[source_name].get(source_relative)
            if source_status is None or source_status.get("status") != "accepted":
                raise RuntimeError(f"approved source is missing accepted review row: {source_relative}")
            combined_relative = destination_image.relative_to(args.output).as_posix()
            combined_rows.append(
                {
                    "relative_image": combined_relative,
                    "split": split,
                    "status": "accepted",
                    "uncertain": source_status.get("uncertain", "False"),
                    "score": source_status.get("score", ""),
                    "peak_gap": source_status.get("peak_gap", ""),
                    "object_expected": source_status.get("object_expected", "1"),
                    "source_dataset": source_name,
                    "source_relative_image": source_relative,
                    "source_kind": source_status.get("source_kind", source_name),
                    "source_frame": source_status.get("source_frame", ""),
                }
            )
            inventory.append(
                {
                    "combined_image": combined_relative,
                    "source_dataset": source_name,
                    "source_image": str(image_path.resolve()),
                    "source_label": str(source_label.resolve()),
                    "image_size": destination_image.stat().st_size,
                    "image_sha256": file_sha256(destination_image),
                    "label_sha256": file_sha256(destination_label),
                }
            )

    with (args.output / "review" / "status.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(combined_rows)

    counts: dict[str, dict[str, int]] = {}
    for row in combined_rows:
        split = row["split"]
        group = counts.setdefault(split, {"images": 0, "positive": 0, "negative": 0})
        group["images"] += 1
        key = "positive" if row["object_expected"] == "1" else "negative"
        group[key] += 1
    inventory_text = json.dumps(inventory, sort_keys=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.output.resolve()),
        "purpose": "base still images plus manually reviewed diagnostic-video adaptation frames",
        "final_test_videos_included": False,
        "source_datasets": {
            "base": {
                "path": str(args.base.resolve()),
                "approval": base_approval,
            },
            "adaptation": {
                "path": str(args.adaptation.resolve()),
                "approval": adaptation_approval,
            },
        },
        "counts": counts,
        "inventory_count": len(inventory),
        "inventory_sha256": hashlib.sha256(inventory_text.encode("utf-8")).hexdigest(),
        "inventory": inventory,
    }
    manifest_path = args.output / "manifests" / "source_inventory.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "inventory"}, indent=2))
    print(f"Combined dataset requires approval: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

