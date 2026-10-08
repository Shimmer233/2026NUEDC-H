from __future__ import annotations
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
for relative, expected in manifest["portable_file_sha256"].items():
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"missing bundle file: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"hash mismatch: {relative}: {actual} != {expected}")
lines = [line.strip() for line in (root / "dataset.txt").read_text().splitlines() if line.strip()]
if len(lines) != manifest["calibration"]["images"]:
    raise SystemExit("dataset.txt image count mismatch")
if any(any(character.isspace() for character in line) for line in lines):
    raise SystemExit("dataset.txt paths must not contain whitespace")
if any(not (root / line).is_file() for line in lines):
    raise SystemExit("dataset.txt contains a missing image")
print(f"bundle verified: {len(lines)} calibration images")
