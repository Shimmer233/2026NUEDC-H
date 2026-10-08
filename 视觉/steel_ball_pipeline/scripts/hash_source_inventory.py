from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a read-only source image inventory.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"D:\University\NUEDC\yolov5-master\yolov5-master\VOCData\images"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"D:\University\NUEDC\datasets\steel_ball_v2\manifests\source_inventory.json"
        ),
    )
    args = parser.parse_args()
    paths = sorted(args.source.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"no source images found: {args.source}")
    combined = hashlib.sha256()
    files: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        digest = file_sha256(path)
        size = path.stat().st_size
        files.append({"name": path.name, "size": size, "sha256": digest})
        combined.update(path.name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\0")
        if index % 200 == 0:
            print(f"hashed {index}/{len(paths)}")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()),
        "read_only_operation": True,
        "file_count": len(files),
        "inventory_sha256": combined.hexdigest(),
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Source inventory: {args.output}")
    print(f"Combined SHA-256: {combined.hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
