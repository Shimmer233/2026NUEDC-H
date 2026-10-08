from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.rknn_postprocess import decode_yolov5_heads  # noqa: E402
from steel_ball.vision import letterbox_black, unletterbox_box  # noqa: E402


def longest_false_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def parse_roi(text: str) -> tuple[int, int, int, int]:
    values = tuple(int(value) for value in text.split(","))
    if len(values) != 4:
        raise ValueError("ROI must be x,y,width,height")
    return values


def parse_size(text: str) -> tuple[int, int]:
    values = tuple(int(value) for value in text.split(","))
    if len(values) != 2:
        raise ValueError("size must be width,height")
    return values


def parse_thresholds(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("at least one confidence threshold is required")
    if any(value <= 0.001 or value > 1.0 for value in values):
        raise ValueError("thresholds must be greater than 0.001 and at most 1.0")
    return tuple(sorted(set(values), reverse=True))


def load_frame_policy(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != 1:
        raise ValueError("frame policy schema_version must be 1")
    if policy.get("default_state", "ignore") not in {"ball", "empty", "ignore"}:
        raise ValueError("frame policy default_state must be ball, empty, or ignore")
    videos = policy.get("videos")
    if not isinstance(videos, dict):
        raise ValueError("frame policy must contain a videos object")
    for kind, rules in videos.items():
        if not isinstance(rules, list):
            raise ValueError(f"frame policy rules for {kind} must be a list")
        occupied: list[tuple[int, int]] = []
        for rule in rules:
            start = int(rule["start"])
            end = int(rule["end"])
            state = str(rule["state"])
            if start < 0 or end < start:
                raise ValueError(f"invalid frame range for {kind}: {start}..{end}")
            if state not in {"ball", "empty", "ignore"}:
                raise ValueError(f"invalid frame state for {kind}: {state}")
            if any(not (end < prior_start or start > prior_end) for prior_start, prior_end in occupied):
                raise ValueError(f"overlapping frame ranges for {kind}: {start}..{end}")
            occupied.append((start, end))
    return policy


def evaluation_state(
    kind: str,
    frame_index: int,
    default_expected_ball: bool,
    policy: dict[str, object] | None,
) -> str:
    default = "ball" if default_expected_ball else "empty"
    if policy is None:
        return default
    rules = policy["videos"].get(kind, [])
    for rule in rules:
        if int(rule["start"]) <= frame_index <= int(rule["end"]):
            return str(rule["state"])
    return str(policy.get("default_state", default))


def record_state(record: dict[str, object]) -> str:
    state = record.get("evaluation_state")
    if state in {"ball", "empty", "ignore"}:
        return str(state)
    expected = record.get("expected_ball")
    if isinstance(expected, str):
        return "ball" if expected.lower() in {"1", "true", "yes"} else "empty"
    return "ball" if bool(expected) else "empty"


def crop_frame(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    x, y, width, height = roi
    crop = frame[y : y + height, x : x + width]
    if crop.shape[:2] != (height, width):
        raise ValueError(f"ROI {roi} exceeds frame shape {frame.shape}")
    return cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)


class OnnxBackend:
    def __init__(self, model: Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model), providers=ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return self.session.run(None, {self.input_name: tensor})


def evaluate_video(
    backend: OnnxBackend,
    video: Path,
    kind: str,
    expected_ball: bool,
    roi: tuple[int, int, int, int],
    output_size: tuple[int, int],
    track_y_min: float,
    track_y_max: float,
    frame_policy: dict[str, object] | None,
) -> tuple[list[dict[str, object]], list[float]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open {video}")
    records: list[dict[str, object]] = []
    latencies: list[float] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        crop = crop_frame(frame, roi, output_size)
        model_input, meta = letterbox_black(crop, 640)
        started = time.perf_counter()
        outputs = backend.infer(model_input)
        latencies.append((time.perf_counter() - started) * 1000)
        detections = decode_yolov5_heads(outputs, confidence_threshold=0.001)
        candidates = []
        for detection in detections:
            mapped_box = unletterbox_box(detection.box, meta)
            center_y = float((mapped_box[1] + mapped_box[3]) / 2)
            if track_y_min <= center_y <= track_y_max:
                candidates.append((detection, mapped_box))
        if candidates:
            detection, box = candidates[0]
            confidence = detection.confidence
            box_values = [float(value) for value in box]
        else:
            confidence = 0.0
            box_values = [None, None, None, None]
        state = evaluation_state(kind, frame_index, expected_ball, frame_policy)
        records.append(
            {
                "video": kind,
                "video_path": str(video.resolve()),
                "frame_id": frame_index,
                "expected_ball": state == "ball" if state != "ignore" else None,
                "evaluation_state": state,
                "confidence": confidence,
                "x1": box_values[0],
                "y1": box_values[1],
                "x2": box_values[2],
                "y2": box_values[3],
            }
        )
        frame_index += 1
        if frame_index % 500 == 0:
            print(f"{kind}: {frame_index} frames")
    capture.release()
    return records, latencies


def render_contact_sheet(
    records: list[dict[str, object]],
    output: Path,
    roi: tuple[int, int, int, int],
    output_size: tuple[int, int],
) -> None:
    if not records:
        return
    tile_width, tile_height = 470, 140
    columns = 4
    rows = int(np.ceil(len(records) / columns))
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 28, dtype=np.uint8)
    captures: dict[str, cv2.VideoCapture] = {}
    try:
        for index, record in enumerate(records):
            video_path = str(record["video_path"])
            capture = captures.setdefault(video_path, cv2.VideoCapture(video_path))
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(record["frame_id"]))
            ok, frame = capture.read()
            if not ok:
                continue
            crop = crop_frame(frame, roi, output_size)
            if record["x1"] is not None:
                box = tuple(round(float(record[key])) for key in ("x1", "y1", "x2", "y2"))
                cv2.rectangle(crop, box[:2], box[2:], (0, 0, 255), 2)
            text = (
                f"{record['video']} f={record['frame_id']} "
                f"conf={float(record['confidence']):.3f}"
            )
            cv2.putText(crop, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(crop, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            row, column = divmod(index, columns)
            top, left = row * tile_height, column * tile_width
            canvas[top : top + output_size[1], left : left + output_size[0]] = crop
    finally:
        for capture in captures.values():
            capture.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def threshold_metrics(records: list[dict[str, object]], threshold: float) -> dict[str, object]:
    report: dict[str, object] = {}
    kinds = sorted({str(record["video"]) for record in records})
    for kind in kinds:
        subset = [record for record in records if record["video"] == kind]
        ball_subset = [record for record in subset if record_state(record) == "ball"]
        empty_subset = [record for record in subset if record_state(record) == "empty"]
        ignored = len(subset) - len(ball_subset) - len(empty_subset)
        if ball_subset and empty_subset:
            raise ValueError(f"video {kind} mixes ball and empty evaluation states")
        if ball_subset:
            detected = [float(record["confidence"]) >= threshold for record in ball_subset]
            longest = current = 0
            for record in subset:
                if record_state(record) != "ball":
                    current = 0
                    continue
                current = 0 if float(record["confidence"]) >= threshold else current + 1
                longest = max(longest, current)
            report[kind] = {
                "expected": "ball",
                "total_video_frames": len(subset),
                "frames": len(ball_subset),
                "ignored_frames": ignored,
                "detected": sum(detected),
                "recall": sum(detected) / len(ball_subset),
                "longest_consecutive_misses": longest,
            }
        elif empty_subset:
            detected = [float(record["confidence"]) >= threshold for record in empty_subset]
            false_positives = sum(detected)
            report[kind] = {
                "expected": "empty",
                "total_video_frames": len(subset),
                "frames": len(empty_subset),
                "ignored_frames": ignored,
                "false_positive_frames": false_positives,
                "false_positives_per_1000_frames": false_positives * 1000 / len(empty_subset),
            }
        else:
            report[kind] = {
                "expected": "ignore",
                "total_video_frames": len(subset),
                "frames": 0,
                "ignored_frames": ignored,
            }
    ball_records = [record for record in records if record_state(record) == "ball"]
    ball_detected = [float(record["confidence"]) >= threshold for record in ball_records]
    empty_records = [record for record in records if record_state(record) == "empty"]
    empty_false_positives = sum(float(record["confidence"]) >= threshold for record in empty_records)
    ball_video_reports = [
        value
        for value in report.values()
        if isinstance(value, dict) and value.get("expected") == "ball"
    ]
    report["combined"] = {
        "ball_frames": len(ball_records),
        "ball_recall": sum(ball_detected) / len(ball_detected) if ball_detected else None,
        "longest_consecutive_misses_per_video": max(
            int(value["longest_consecutive_misses"])
            for value in ball_video_reports
        ) if ball_video_reports else None,
        "empty_frames": len(empty_records),
        "empty_false_positive_frames": empty_false_positives,
        "empty_false_positives_per_1000_frames": (
            empty_false_positives * 1000 / len(empty_records) if empty_records else None
        ),
        "ignored_frames": sum(record_state(record) == "ignore" for record in records),
    }
    combined = report["combined"]
    report["passes"] = bool(
        ball_video_reports
        and empty_records
        and all(float(value["recall"]) >= 0.99 for value in ball_video_reports)
        and all(int(value["longest_consecutive_misses"]) <= 3 for value in ball_video_reports)
        and float(combined["empty_false_positives_per_1000_frames"]) <= 1.0
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frame recall and empty-track false positives.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--fast", type=Path, required=True)
    parser.add_argument("--empty", type=Path, required=True)
    parser.add_argument("--roi", default="170,190,1240,290")
    parser.add_argument("--output-size", default="470,110")
    parser.add_argument("--track-y-min", type=float, default=25.0)
    parser.add_argument("--track-y-max", type=float, default=85.0)
    parser.add_argument("--frame-policy", type=Path)
    parser.add_argument(
        "--thresholds",
        default="0.25,0.20,0.15,0.10,0.05",
        help="comma-separated confidence thresholds, evaluated from highest to lowest",
    )
    parser.add_argument("--purpose", choices=("acceptance", "calibration"), default="acceptance")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roi = parse_roi(args.roi)
    output_size = parse_size(args.output_size)
    thresholds = parse_thresholds(args.thresholds)
    frame_policy = load_frame_policy(args.frame_policy)
    backend = OnnxBackend(args.model)
    all_records: list[dict[str, object]] = []
    all_latencies: list[float] = []
    for video, kind, expected in (
        (args.normal, "normal", True),
        (args.fast, "fast", True),
        (args.empty, "empty", False),
    ):
        records, latencies = evaluate_video(
            backend,
            video,
            kind,
            expected,
            roi,
            output_size,
            args.track_y_min,
            args.track_y_max,
            frame_policy,
        )
        all_records.extend(records)
        all_latencies.extend(latencies)
    args.output.mkdir(parents=True, exist_ok=True)
    fields = (
        "video", "video_path", "frame_id", "expected_ball", "evaluation_state",
        "confidence", "x1", "y1", "x2", "y2",
    )
    with (args.output / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)
    threshold_key = lambda value: f"{value:.6f}".rstrip("0").rstrip(".")
    metrics = {threshold_key(threshold): threshold_metrics(all_records, threshold) for threshold in thresholds}
    selected = next(
        (threshold for threshold in thresholds if metrics[threshold_key(threshold)]["passes"]),
        None,
    )
    latency_values = np.asarray(all_latencies)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": args.purpose,
        "model": str(args.model.resolve()),
        "roi": roi,
        "output_size": output_size,
        "track_y_band": [args.track_y_min, args.track_y_max],
        "total_frames": len(all_records),
        "latency_ms_median_onnx_host": float(np.median(latency_values)),
        "latency_ms_p95_onnx_host": float(np.percentile(latency_values, 95)),
        "thresholds": metrics,
        "selected_threshold": selected,
        "independent_acceptance_passed": bool(args.purpose == "acceptance" and selected is not None),
    }
    if args.frame_policy is not None:
        policy_bytes = args.frame_policy.read_bytes()
        report["frame_policy"] = {
            "path": str(args.frame_policy.resolve()),
            "sha256": hashlib.sha256(policy_bytes).hexdigest(),
            "content": frame_policy,
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ball_records = [record for record in all_records if record_state(record) == "ball"]
    empty_records = [record for record in all_records if record_state(record) == "empty"]
    worst_ball = sorted(ball_records, key=lambda record: float(record["confidence"]))[:32]
    top_empty = sorted(empty_records, key=lambda record: float(record["confidence"]), reverse=True)[:32]
    render_contact_sheet(worst_ball, args.output / "worst_ball_frames.jpg", roi, output_size)
    render_contact_sheet(top_empty, args.output / "top_empty_detections.jpg", roi, output_size)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
