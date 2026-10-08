from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    import resource
except ImportError:  # Windows development host; the board runs Linux.
    resource = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.calibration import TrackCalibration  # noqa: E402
from steel_ball.camera_runtime import LatestFrameCamera, initialize_rknn  # noqa: E402
from steel_ball.tracking import PositionVelocityKalman  # noqa: E402
from steel_ball.vision_display import draw_overlay  # noqa: E402
from steel_ball.vision_runtime import (  # noqa: E402
    AsyncYoloWorker,
    TrackGeometry,
    TrackStabilizer,
)


TRAJECTORY_FIELDS = (
    "timestamp_ns",
    "frame_id",
    "detected",
    "confidence",
    "x_px",
    "y_px",
    "s_cm",
    "velocity_cm_s",
    "predicted",
)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_optional_calibration(path: Path) -> TrackCalibration | None:
    if path.is_file():
        return TrackCalibration.load(path)
    print(
        f"Calibration not found: {path}. Detection will run without cm coordinates.",
        file=sys.stderr,
    )
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="YOLO-only steel-ball vision tracker for Orange Pi 5B."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "vision.yaml"
    )
    parser.add_argument("--camera-device")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args(argv)
    if args.duration < 0:
        raise ValueError("duration cannot be negative")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.camera_device:
        config["camera"]["device"] = args.camera_device
    model_path = args.model.resolve() if args.model else project_path(config["model"])
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    calibration = load_optional_calibration(project_path(config["calibration"]))
    geometry_path = project_path(config.get("geometry", "config/track_geometry.yaml"))
    geometry = TrackGeometry.from_config(config["roi"], geometry_path)
    if not geometry_path.is_file():
        print(
            f"Geometry not found: {geometry_path}. Using rectangular ROI fallback.",
            file=sys.stderr,
        )
    stabilizer = TrackStabilizer(
        config.get("stabilization", {}),
        (geometry.output_width, geometry.output_height),
    )
    tracker = PositionVelocityKalman(
        process_acceleration_std=float(config["tracking"]["process_acceleration_std"]),
        measurement_std=float(config["tracking"]["measurement_std_cm"]),
        max_prediction_frames=int(config["tracking"]["max_prediction_frames"]),
    )
    yolo_max_age_ms = float(config["fusion"].get("yolo_max_age_ms", 80.0))
    yolo_measurement_std_cm = float(
        config["fusion"].get("yolo_measurement_std_cm", 0.12)
    )

    trajectory_path = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else project_path(config["output"]["csv"])
    )
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    display = bool(config["output"].get("display", True)) and not args.headless
    window_name = "Steel ball vision"

    runtime = initialize_rknn(model_path)
    yolo_worker: AsyncYoloWorker | None = None
    camera: LatestFrameCamera | None = None
    trail: deque[tuple[int, int]] = deque(maxlen=160)
    frame_times: deque[float] = deque(maxlen=160)
    yolo_completion_times: deque[float] = deque(maxlen=80)
    processing_intervals: list[float] = []
    latencies: list[float] = []
    rss_samples_mb: list[float] = []
    previous_processed_at: float | None = None
    last_camera_frame = -1
    last_processed_frame = -1
    last_yolo_result_id = -1
    last_yolo_completion_ns = 0
    last_yolo_error: str | None = None
    next_process_ns = 0
    next_display_time = 0.0
    fusion_period_ns = round(1e9 / float(config["runtime"]["fusion_hz"]))
    display_period = 1.0 / float(config["runtime"]["display_hz"])
    start = time.monotonic()

    try:
        yolo_worker = AsyncYoloWorker(
            runtime, {**config["detection"], **config.get("yolo", {})}
        )
        camera = LatestFrameCamera(config["camera"])
        if display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        with trajectory_path.open(
            "w", newline="", encoding="utf-8", buffering=1
        ) as trajectory_stream:
            writer = csv.DictWriter(trajectory_stream, fieldnames=TRAJECTORY_FIELDS)
            writer.writeheader()
            while not args.duration or time.monotonic() - start < args.duration:
                packet = camera.next(last_camera_frame)
                if packet is None:
                    continue
                frame_id, timestamp_ns, frame = packet
                last_camera_frame = frame_id
                if next_process_ns == 0:
                    next_process_ns = timestamp_ns
                if timestamp_ns < next_process_ns:
                    continue
                while next_process_ns <= timestamp_ns:
                    next_process_ns += fusion_period_ns
                last_processed_frame = frame_id
                processing_start = time.perf_counter()

                rectified = stabilizer.apply(geometry.warp(frame))
                # Current deployed behavior always requests the 50 Hz reacquire rate.
                yolo_worker.submit(frame_id, timestamp_ns, rectified, True)
                yolo_result = yolo_worker.latest()
                if (
                    yolo_result is not None
                    and yolo_result.completed_ns != last_yolo_completion_ns
                ):
                    last_yolo_completion_ns = yolo_result.completed_ns
                    yolo_completion_times.append(time.monotonic())
                if (
                    yolo_result is not None
                    and yolo_result.error is not None
                    and yolo_result.error != last_yolo_error
                ):
                    print(f"YOLO error: {yolo_result.error}", file=sys.stderr)
                    last_yolo_error = yolo_result.error

                yolo_is_new = (
                    yolo_result is not None
                    and yolo_result.frame_id != last_yolo_result_id
                )
                yolo_age_ms = (
                    float("inf")
                    if yolo_result is None
                    else max(0.0, (timestamp_ns - yolo_result.timestamp_ns) / 1e6)
                )
                candidate = (
                    yolo_result.candidate
                    if yolo_result is not None
                    and yolo_result.error is None
                    and yolo_age_ms <= yolo_max_age_ms
                    else None
                )
                measurement_candidate = candidate if yolo_is_new else None
                if yolo_is_new and yolo_result is not None:
                    last_yolo_result_id = yolo_result.frame_id

                measurement = None
                frame_center: tuple[float, float] | None = None
                frame_box: np.ndarray | None = None
                if candidate is not None:
                    frame_x, frame_y = geometry.rectified_to_frame(
                        candidate.center_x, candidate.center_y
                    )
                    frame_center = (frame_x, frame_y)
                    frame_box = geometry.box_to_frame(candidate.box)
                if measurement_candidate is not None and calibration is not None:
                    measurement = calibration.position_cm(frame_x, frame_y)
                    if yolo_result is not None and tracker.state is not None:
                        age_seconds = max(
                            0.0, (timestamp_ns - yolo_result.timestamp_ns) / 1e9
                        )
                        measurement += float(tracker.state[1]) * age_seconds

                estimate = tracker.step(
                    measurement,
                    timestamp_ns / 1e9,
                    measurement_std=yolo_measurement_std_cm,
                    innovation_gate_sigma=float(
                        config["fusion"]["innovation_gate_sigma"]
                    ),
                )
                if frame_center is not None and (
                    estimate.detected or (calibration is None and yolo_is_new)
                ):
                    trail.append(
                        (round(frame_center[0]), round(frame_center[1]))
                    )

                confidence = candidate.confidence if candidate is not None else None
                writer.writerow(
                    {
                        "timestamp_ns": timestamp_ns,
                        "frame_id": frame_id,
                        "detected": int(
                            estimate.detected
                            if calibration is not None
                            else candidate is not None
                        ),
                        "confidence": (
                            "" if confidence is None else f"{confidence:.6f}"
                        ),
                        "x_px": (
                            "" if frame_center is None else f"{frame_center[0]:.3f}"
                        ),
                        "y_px": (
                            "" if frame_center is None else f"{frame_center[1]:.3f}"
                        ),
                        "s_cm": (
                            ""
                            if estimate.position_cm is None
                            else f"{estimate.position_cm:.5f}"
                        ),
                        "velocity_cm_s": (
                            ""
                            if estimate.velocity_cm_s is None
                            else f"{estimate.velocity_cm_s:.5f}"
                        ),
                        "predicted": int(estimate.predicted),
                    }
                )

                latency_ms = (time.perf_counter() - processing_start) * 1000.0
                latencies.append(latency_ms)
                processed_at = time.monotonic()
                if previous_processed_at is not None:
                    processing_intervals.append(processed_at - previous_processed_at)
                previous_processed_at = processed_at
                frame_times.append(processed_at)
                if resource is not None and (
                    not rss_samples_mb or len(latencies) % 240 == 0
                ):
                    rss_samples_mb.append(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                    )
                loop_fps = (
                    (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                    if len(frame_times) >= 2
                    else 0.0
                )
                yolo_fps = (
                    (len(yolo_completion_times) - 1)
                    / (yolo_completion_times[-1] - yolo_completion_times[0])
                    if len(yolo_completion_times) >= 2
                    else 0.0
                )
                if display and processed_at >= next_display_time:
                    next_display_time = processed_at + display_period
                    draw_overlay(
                        frame,
                        geometry,
                        candidate,
                        frame_box,
                        frame_center,
                        trail,
                        estimate.position_cm,
                        estimate.velocity_cm_s,
                        estimate.predicted,
                        calibration,
                        config.get("coordinate", {}),
                        latency_ms,
                        loop_fps,
                        yolo_fps,
                    )
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None:
            camera.close()
        if yolo_worker is not None:
            yolo_worker.close()
        runtime.release()
        cv2.destroyAllWindows()

    interval_values = np.asarray(processing_intervals, dtype=np.float64)
    latency_values = np.asarray(latencies, dtype=np.float64)
    minimum_fps = float(config["runtime"].get("minimum_accepted_fps", 50.0))
    median_fps = (
        float(np.median(1.0 / interval_values)) if interval_values.size else 0.0
    )
    performance = {
        "processed_frames": len(latencies),
        "last_camera_frame_id": last_camera_frame,
        "last_processed_frame_id": last_processed_frame,
        "duration_seconds": time.monotonic() - start,
        "fusion_fps_median": median_fps,
        "configured_fusion_hz": float(config["runtime"]["fusion_hz"]),
        "minimum_accepted_fps": minimum_fps,
        "latency_ms_p95": (
            float(np.percentile(latency_values, 95))
            if latency_values.size
            else None
        ),
        "rss_max_mb": max(rss_samples_mb) if rss_samples_mb else None,
        "rss_growth_mb": (
            rss_samples_mb[-1] - rss_samples_mb[0]
            if len(rss_samples_mb) >= 2
            else None
        ),
        "passes_minimum_fps": bool(interval_values.size and median_fps >= minimum_fps),
    }
    performance_path = trajectory_path.with_suffix(".performance.json")
    performance_path.write_text(
        json.dumps(performance, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(performance, indent=2))
    print(f"Trajectory CSV: {trajectory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
