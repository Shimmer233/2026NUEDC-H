from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SystemExit(f"manifest path escapes its root: {relative}") from error
    return path


def verify_file(path: Path, relative: str, record: dict[str, object]) -> None:
    if not path.is_file():
        raise SystemExit(f"file is missing: {relative}")
    if path.stat().st_size != int(record["size"]):
        raise SystemExit(f"file size mismatch: {relative}")
    if sha256(path) != record["sha256"]:
        raise SystemExit(f"file hash mismatch: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a complete models/config backup.")
    parser.add_argument("backup_dir", type=Path)
    parser.add_argument("--restored-root", type=Path)
    args = parser.parse_args()

    backup = args.backup_dir.resolve()
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, dict) or not records:
        raise SystemExit("backup manifest has no files")
    for relative, record in records.items():
        verify_file(checked_path(backup, relative), relative, record)

    if args.restored_root is not None:
        restored = args.restored_root.resolve()
        for relative, record in records.items():
            verify_file(checked_path(restored, relative), relative, record)
        print(f"Restored models/config verified: {len(records)} files")
    else:
        print(f"Backup verified: {len(records)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
