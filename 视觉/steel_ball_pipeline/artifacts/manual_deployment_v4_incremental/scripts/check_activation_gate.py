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
    parser = argparse.ArgumentParser(
        description="Require passing normal, fast, and empty runs for the current staged model."
    )
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    args = parser.parse_args()

    if not args.stage_manifest.is_file():
        raise FileNotFoundError(args.stage_manifest)
    stage = json.loads(args.stage_manifest.read_text(encoding="utf-8"))
    stage_sha256 = sha256(args.stage_manifest)
    stage_files = stage.get("files", {})
    expected_int8 = stage_files.get(
        "models/steel_ball_yolov5n_v4_incremental_int8.rknn", {}
    ).get("sha256")
    expected_config = stage_files.get(
        "config/vision_v4_incremental_staged.yaml", {}
    ).get("sha256")
    if not expected_int8 or not expected_config:
        raise SystemExit("stage manifest is missing the staged model or config hash")

    attempts: dict[str, list[tuple[Path, dict[str, object]]]] = {
        "normal": [],
        "fast": [],
        "empty": [],
    }
    if args.runs_root.is_dir():
        for path in args.runs_root.glob("*/attempt.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            scenario = payload.get("scenario")
            if scenario in attempts:
                attempts[str(scenario)].append((path, payload))

    selected: dict[str, dict[str, object]] = {}
    errors = []
    for scenario, records in attempts.items():
        if not records:
            errors.append(f"missing {scenario} test")
            continue
        attempt_path, attempt = max(
            records, key=lambda item: item[0].parent.stat().st_mtime_ns
        )
        if attempt.get("stage_manifest_sha256") != stage_sha256:
            errors.append(f"latest {scenario} attempt belongs to another staging operation")
        if attempt.get("camera") != args.camera:
            errors.append(f"latest {scenario} attempt used another camera")
        path = attempt_path.parent / "summary.json"
        if not path.is_file():
            errors.append(f"latest {scenario} attempt is incomplete")
            selected[scenario] = {"attempt": str(attempt_path), **attempt}
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"latest {scenario} summary is unreadable")
            selected[scenario] = {"attempt": str(attempt_path), **attempt}
            continue
        selected[scenario] = {"summary": str(path), **payload}
        if payload.get("scenario") != scenario:
            errors.append(f"latest {scenario} summary has the wrong scenario identity")
        if payload.get("camera") != args.camera:
            errors.append(f"latest {scenario} summary used another camera")
        if payload.get("stage_manifest_sha256") != stage_sha256:
            errors.append(f"latest {scenario} test belongs to another staging operation")
        if payload.get("staged_int8_sha256") != expected_int8:
            errors.append(f"latest {scenario} test used another INT8 model")
        if payload.get("staged_config_sha256") != expected_config:
            errors.append(f"latest {scenario} test used another staged config")
        if not payload.get("passed"):
            errors.append(f"latest {scenario} test did not pass")
        if float(payload.get("actual_duration_seconds", 0.0)) < 55.0:
            errors.append(f"latest {scenario} test is shorter than 55 seconds")
        if not payload.get("frame_count_gate_passed"):
            errors.append(f"latest {scenario} test has inconsistent frame counts")
        if not payload.get("fusion_fps_gate_passed"):
            errors.append(f"latest {scenario} test is below the minimum fusion FPS")
        if payload.get("log_errors"):
            errors.append(f"latest {scenario} test contains runtime errors")

    report = {
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": stage_sha256,
        "selected": selected,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit("activation gate is closed: " + "; ".join(errors))
    print("Activation gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
