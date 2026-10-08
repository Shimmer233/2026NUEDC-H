from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


ERROR_PATTERN = re.compile(r"Traceback|YOLO inference warning|RuntimeError")
MINIMUM_FUSION_FPS = 30.0
MINIMUM_ACCEPTED_MEASUREMENTS_HZ = 30.0
REQUIRED_FIELDS = {
    "timestamp_ns",
    "frame_id",
    "detected",
    "confidence",
    "x_px",
    "y_px",
    "s_cm",
    "velocity_cm_s",
    "predicted",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def longest_false_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def flag(row: dict[str, str], field: str) -> bool:
    return row.get(field, "").strip().lower() in {"1", "true", "yes"}


def present(row: dict[str, str], field: str) -> bool:
    return bool(row.get(field, "").strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize one staged Orange Pi test run.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--scenario", choices=("normal", "fast", "empty"), required=True)
    parser.add_argument("--requested-duration", type=float, required=True)
    args = parser.parse_args()

    for path in (args.csv, args.log, args.stage_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    with args.csv.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED_FIELDS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_FIELDS.difference(reader.fieldnames or []))
            raise RuntimeError(f"trajectory CSV is missing fields: {missing}")
        rows = list(reader)

    performance_path = args.csv.with_suffix(".performance.json")
    if not performance_path.is_file():
        raise FileNotFoundError(performance_path)
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    stage = json.loads(args.stage_manifest.read_text(encoding="utf-8"))
    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    log_errors = [
        f"line {index}: {line[:500]}"
        for index, line in enumerate(log_text.splitlines(), start=1)
        if ERROR_PATTERN.search(line)
    ][:20]

    actual_duration = float(performance.get("duration_seconds", 0.0))
    processed_frames = int(performance.get("processed_frames", 0))
    fusion_fps = float(performance.get("fusion_fps_median", 0.0))
    average_fps = processed_frames / actual_duration if actual_duration > 0.0 else 0.0
    duration_ok = actual_duration >= 55.0 and args.requested_duration >= 55.0
    frame_count_ok = bool(rows and processed_frames == len(rows))
    fps_ok = fusion_fps >= MINIMUM_FUSION_FPS and average_fps >= MINIMUM_FUSION_FPS
    candidate_present = [present(row, "confidence") for row in rows]
    detected = [flag(row, "detected") for row in rows]
    predicted = [flag(row, "predicted") for row in rows]
    position_present = [present(row, "s_cm") for row in rows]
    coordinate_present = [
        present(row, "x_px") or present(row, "y_px") for row in rows
    ]
    velocity_present = [present(row, "velocity_cm_s") for row in rows]

    stage_files = stage.get("files", {})
    payload: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": args.scenario,
        "camera": args.camera,
        "trajectory": str(args.csv.resolve()),
        "run_log": str(args.log.resolve()),
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": sha256(args.stage_manifest),
        "staged_int8_sha256": stage_files.get(
            "models/steel_ball_yolov5n_v4_incremental_int8.rknn", {}
        ).get("sha256"),
        "staged_config_sha256": stage_files.get(
            "config/vision_v4_incremental_staged.yaml", {}
        ).get("sha256"),
        "requested_duration_seconds": args.requested_duration,
        "actual_duration_seconds": actual_duration,
        "frames": len(rows),
        "processed_frames": processed_frames,
        "duration_gate_passed": duration_ok,
        "frame_count_gate_passed": frame_count_ok,
        "minimum_fusion_fps": MINIMUM_FUSION_FPS,
        "average_fps": average_fps,
        "fusion_fps_gate_passed": fps_ok,
        "log_errors": log_errors,
        "log_gate_passed": not log_errors,
        "accepted_measurement_frames": sum(detected),
        "predicted_frames": sum(predicted),
        "position_frames": sum(position_present),
        "performance": performance,
    }
    if args.scenario in {"normal", "fast"}:
        candidate_frames = sum(candidate_present)
        recall_proxy = candidate_frames / len(rows) if rows else 0.0
        longest_misses = longest_false_run(candidate_present)
        position_frames = sum(position_present)
        position_availability = position_frames / len(rows) if rows else 0.0
        longest_position_gaps = longest_false_run(position_present)
        accepted_measurement_hz = (
            sum(detected) / actual_duration if actual_duration > 0.0 else 0.0
        )
        passed = bool(
            duration_ok
            and frame_count_ok
            and fps_ok
            and not log_errors
            and recall_proxy >= 0.99
            and longest_misses <= 3
            and position_availability >= 0.99
            and longest_position_gaps <= 3
            and accepted_measurement_hz >= MINIMUM_ACCEPTED_MEASUREMENTS_HZ
        )
        payload.update(
            {
                "expected": "ball",
                "detected_frames": candidate_frames,
                "recall": recall_proxy,
                "recall_metric": (
                    "confidence_present_per_fusion_sample; proxy because the current "
                    "CSV does not identify individual inference completions"
                ),
                "longest_consecutive_misses": longest_misses,
                "position_availability": position_availability,
                "longest_consecutive_position_gaps": longest_position_gaps,
                "accepted_measurement_hz": accepted_measurement_hz,
                "minimum_accepted_measurements_hz": MINIMUM_ACCEPTED_MEASUREMENTS_HZ,
                "passed": passed,
            }
        )
    else:
        false_positives = sum(candidate_present)
        per_1000 = false_positives * 1000.0 / len(rows) if rows else float("inf")
        tracked_detections = sum(detected)
        tracked_predictions = sum(predicted)
        positions = sum(position_present)
        passed = bool(
            duration_ok
            and frame_count_ok
            and fps_ok
            and not log_errors
            and rows
            and per_1000 <= 1.0
            and tracked_detections == 0
            and tracked_predictions == 0
            and positions == 0
        )
        payload.update(
            {
                "expected": "empty",
                "false_positive_frames": false_positives,
                "false_positives_per_1000_frames": per_1000,
                "tracked_detection_frames": tracked_detections,
                "tracked_prediction_frames": tracked_predictions,
                "position_output_frames": positions,
                "coordinate_output_frames": sum(coordinate_present),
                "velocity_output_frames": sum(velocity_present),
                "passed": passed,
            }
        )

    output = args.csv.parent / "summary.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2))
    print(f"Summary: {output}")
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
