from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.dataset import (  # noqa: E402
    APPROVAL_FILE,
    image_files,
    label_for_image,
    labels_digest,
    read_review_rows,
    validate_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approve a fully reviewed dataset for training.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\University\NUEDC\datasets\steel_ball_v2"),
    )
    parser.add_argument("--confirm", action="store_true", help="Write the approval marker after validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = image_files(args.dataset, ("train", "val"))
    review_rows = read_review_rows(args.dataset)
    errors: list[str] = []
    negative_count = 0
    for image_path in images:
        relative = image_path.relative_to(args.dataset).as_posix()
        row = review_rows.get(relative)
        if row is None or row.get("status") != "accepted":
            errors.append(f"not accepted: {relative}")
        require_one = row is None or row.get("object_expected", "1") == "1"
        negative_count += int(not require_one)
        label_errors = validate_label(label_for_image(args.dataset, image_path), require_one=require_one)
        errors.extend(f"{relative}: {error}" for error in label_errors)
    if errors:
        print(f"Dataset is not approved: {len(errors)} error(s).")
        for error in errors[:50]:
            print(f"  {error}")
        return 1
    if not args.confirm:
        print(f"Validation passed for {len(images)} images. Re-run with --confirm to approve.")
        return 0
    payload = {
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "image_count": len(images),
        "positive_image_count": len(images) - negative_count,
        "negative_image_count": negative_count,
        "labels_sha256": labels_digest(args.dataset),
    }
    approval_path = args.dataset / APPROVAL_FILE
    approval_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Dataset approved: {approval_path}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
