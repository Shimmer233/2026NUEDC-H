from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_BASE_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0"
MODELS = ("yolov5n", "yolov5s")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official Ultralytics YOLOv5 v7 weights.")
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    output_dir = PROJECT_ROOT / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model}.pt"
    url = f"{OFFICIAL_BASE_URL}/{args.model}.pt"
    if not output_path.exists():
        temporary = output_path.with_suffix(".download")
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(output_path)
    record = {
        "model": args.model,
        "source": url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "size": output_path.stat().st_size,
        "sha256": sha256(output_path),
    }
    manifest_path = output_dir / f"{args.model}.json"
    manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
