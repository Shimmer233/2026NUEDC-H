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


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify every staged deployment file.")
    parser.add_argument("app_dir", type=Path)
    parser.add_argument("stage_manifest", type=Path)
    args = parser.parse_args()

    app = args.app_dir.resolve()
    manifest_path = args.stage_manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("active_model_changed") is not False:
        raise SystemExit("invalid stage manifest: active_model_changed must be false")
    source_config = app / "config" / "vision.yaml"
    expected_source_hash = payload.get("source_config_sha256")
    if not expected_source_hash:
        raise SystemExit("invalid stage manifest: source config hash is missing")
    if not source_config.is_file() or sha256(source_config) != expected_source_hash:
        raise SystemExit(
            "active vision.yaml changed after staging; stage and test the model again"
        )
    records = payload.get("files")
    if not isinstance(records, dict) or not records:
        raise SystemExit("invalid stage manifest: files are missing")

    for relative, record in records.items():
        path = (app / relative).resolve()
        try:
            path.relative_to(app)
        except ValueError as error:
            raise SystemExit(f"staged path escapes application directory: {relative}") from error
        if not path.is_file():
            raise SystemExit(f"staged file is missing: {relative}")
        if path.stat().st_size != int(record["size"]):
            raise SystemExit(f"staged file size mismatch: {relative}")
        if sha256(path) != record["sha256"]:
            raise SystemExit(f"staged file hash mismatch: {relative}")

    dependencies = payload.get("runtime_dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        raise SystemExit("invalid stage manifest: runtime dependencies are missing")
    for relative, record in dependencies.items():
        path = (app / relative).resolve()
        try:
            path.relative_to(app)
        except ValueError as error:
            raise SystemExit(f"runtime dependency escapes application directory: {relative}") from error
        if not path.is_file():
            raise SystemExit(f"runtime dependency is missing: {relative}")
        if path.stat().st_size != int(record["size"]) or sha256(path) != record["sha256"]:
            raise SystemExit(
                f"runtime dependency changed after staging: {relative}; stage and test again"
            )

    print(
        f"Staged deployment verified: {len(records)} files, "
        f"{len(dependencies)} runtime dependencies"
    )
    print(f"Stage manifest SHA-256: {sha256(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
