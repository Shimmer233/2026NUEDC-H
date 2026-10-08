from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Hash a complete models/config backup.")
    parser.add_argument("backup_dir", type=Path)
    parser.add_argument("--purpose", required=True)
    args = parser.parse_args()

    root = args.backup_dir.resolve()
    for name in ("models", "config"):
        if not (root / name).is_dir():
            raise SystemExit(f"backup is missing the {name} directory")
    files = sorted(
        path
        for name in ("models", "config")
        for path in (root / name).rglob("*")
        if path.is_file()
    )
    if not files:
        raise SystemExit("backup contains no files")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": args.purpose,
        "files": {
            path.relative_to(root).as_posix(): {
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    output = root / "manifest.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"Backup manifest written: {output} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
