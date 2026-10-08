from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    import resource
except ImportError:  # Windows development host; the target board runs Linux.
    resource = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.calibration import TrackCalibration  # noqa: E402
from steel_ball.camera_controls import apply_saved_camera_controls  # noqa: E402
from steel_ball.detector import (  # noqa: E402
    AsyncRoiYoloWorker,
    AsyncYoloWorker,
    BallCandidate,
    TrackGeometry,
    TrackStabilizer,
)
from steel_ball.tracking import PositionVelocityKalman  # noqa: E402


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


class LatestFrameCamera:
    """Keep only the newest camera frame so processing never builds a queue."""

    def __init__(self, config: dict) -> None:
        source_value = config.get("device", config.get("index", 0))
        source = int(source_value) if str(source_value).isdigit() else str(source_value)
        self.capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        fourcc = str(config.get("fourcc", "MJPG"))
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["width"]))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["height"]))
        self.capture.set(cv2.CAP_PROP_FPS, int(config["fps"]))
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            raise RuntimeError(f"unable to open V4L2 camera {source_value}")

        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        expected_size = (int(config["width"]), int(config["height"]))
        if (actual_width, actual_height) != expected_size:
            self.capture.release()
            raise RuntimeError(
                f"camera rejected {expected_size[0]}x{expected_size[1]}; "
                f"actual mode is {actual_width}x{actual_height}@{actual_fps:.3f}"
            )

        control_device = (
            str(source_value)
            if str(source_value).startswith("/dev/")
            else f"/dev/video{source}"
        )
        try:
            applied = apply_saved_camera_controls(
                control_device, config.get("controls", {})
            )
        except RuntimeError as error:
            print(f"Camera control warning: {error}", file=sys.stderr)
            applied = {}
        print(
            f"Camera {source_value}: {actual_width}x{actual_height}@{actual_fps:.3f}, "
            f"fourcc={fourcc}"
        )
        if applied:
            print(f"Camera controls: {applied}")

        self.condition = threading.Condition()
        self.frame: np.ndarray | None = None
        self.frame_id = -1
        self.timestamp_ns = 0
        self.stopped = False
        self.thread = threading.Thread(
            target=self._reader, name="camera-reader", daemon=True
        )
        self.thread.start()

    def _reader(self) -> None:
        while not self.stopped:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.002)
                continue
            with self.condition:
                self.frame = frame
                self.frame_id += 1
                self.timestamp_ns = time.time_ns()
                self.condition.notify_all()

    def next(
        self, after_id: int, timeout: float = 1.0
    ) -> tuple[int, int, np.ndarray] | None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.stopped and self.frame_id <= after_id:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            if self.frame is None:
                return None
            return self.frame_id, self.timestamp_ns, self.frame.copy()

    def close(self) -> None:
        self.stopped = True
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2.0)
        self.capture.release()


def initialize_rknn(model_path: Path):
    try:
        from rknnlite.api import RKNNLite
    except ImportError as error:
        raise RuntimeError(
            "install Rockchip rknn_toolkit_lite2 2.3.2 for RK3588/aarch64"
        ) from error
    runtime = RKNNLite(verbose=False)
    if runtime.load_rknn(str(model_path)) != 0:
        raise RuntimeError(f"unable to load RKNN model: {model_path}")
    core_mask = getattr(RKNNLite, "NPU_CORE_0_1_2", RKNNLite.NPU_CORE_AUTO)
    if runtime.init_runtime(core_mask=core_mask) != 0:
        runtime.release()
        raise RuntimeError("RKNN runtime initialization failed")
    return runtime


def calibration_marker(
    calibration: TrackCalibration | None, position_cm: float | None
) -> tuple[int, int] | None:
    if calibration is None or position_cm is None or not calibration.points:
        return None
    points = sorted(calibration.points, key=lambda item: item[2])
    positions = np.asarray([item[2] for item in points], dtype=np.float64)
    xs = np.asarray([item[0] for item in points], dtype=np.float64)
    ys = np.asarray([item[1] for item in points], dtype=np.float64)
    return (
        round(float(np.interp(position_cm, positions, xs))),
        round(float(np.interp(position_cm, positions, ys))),
    )


def centered_coordinate_cm(position_cm: float, center_cm: float = 12.5) -> float:
    return float(position_cm - center_cm)


def uses_deployment_v3_center(config: dict) -> bool:
    mode = str(config.get("center_detection", {}).get("mode", "rectified")).lower()
    return mode in {"deployment_v3_roi", "direct_roi", "roi"}


def draw_outlined_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_coordinate_axis(
    frame: np.ndarray,
    geometry: TrackGeometry,
    calibration: TrackCalibration | None,
    position_cm: float | None,
    config: dict,
) -> None:
    if calibration is None:
        return
    length_cm = float(config.get("track_length_cm", 25.0))
    center_cm = float(config.get("center_cm", length_cm / 2.0))
    step_cm = float(config.get("minor_tick_cm", 0.5))
    axis_offset_px = float(config.get("axis_offset_px", 18.0))
    if length_cm <= 0 or step_cm <= 0:
        raise ValueError("coordinate dimensions must be positive")
    tick_count = round(length_cm / step_cm)
    if abs(tick_count * step_cm - length_cm) > 1e-6:
        raise ValueError("coordinate step must divide the track length")

    start_value = calibration_marker(calibration, 0.0)
    end_value = calibration_marker(calibration, length_cm)
    if start_value is None or end_value is None:
        return
    start = np.asarray(start_value, dtype=np.float64)
    end = np.asarray(end_value, dtype=np.float64)
    direction = end - start
    distance = float(np.linalg.norm(direction))
    if distance < 20.0:
        return
    direction /= distance
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    bottom_midpoint = np.mean(geometry.source_points[[2, 3]], axis=0)
    if float((bottom_midpoint - (start + end) / 2.0) @ normal) < 0:
        normal = -normal
    axis_start = start + normal * axis_offset_px
    axis_end = end + normal * axis_offset_px
    cv2.arrowedLine(
        frame,
        tuple(np.round(axis_start).astype(int)),
        tuple(np.round(axis_end + direction * 12.0).astype(int)),
        (40, 40, 40),
        2,
        cv2.LINE_AA,
        tipLength=0.025,
    )

    label_interval = int(config.get("label_interval_cm", 2))
    for index in range(tick_count + 1):
        absolute_cm = index * step_cm
        coordinate_cm = absolute_cm - center_cm
        point_value = calibration_marker(calibration, absolute_cm)
        if point_value is None:
            continue
        point = np.asarray(point_value, dtype=np.float64) + normal * axis_offset_px
        is_major = abs(coordinate_cm - round(coordinate_cm)) < 1e-6
        tick_length = 8.0 if is_major else 4.0
        first = tuple(np.round(point - normal * tick_length / 2.0).astype(int))
        second = tuple(np.round(point + normal * tick_length / 2.0).astype(int))
        cv2.line(frame, first, second, (40, 40, 40), 1, cv2.LINE_AA)
        rounded = int(round(coordinate_cm))
        if is_major and label_interval > 0 and rounded % label_interval == 0:
            text = "0" if rounded == 0 else str(rounded)
            location = point + normal * 17.0 - direction * (4.0 * len(text))
            draw_outlined_text(
                frame,
                text,
                tuple(np.round(location).astype(int)),
                0.38,
                (255, 255, 255),
            )

    unit_location = axis_end + direction * 14.0 + normal * 18.0
    draw_outlined_text(
        frame,
        "x/cm",
        tuple(np.round(unit_location).astype(int)),
        0.42,
        (255, 255, 255),
    )
    marker = calibration_marker(calibration, position_cm)
    if marker is not None and position_cm is not None:
        ball_point = np.asarray(marker, dtype=np.float64)
        axis_point = ball_point + normal * axis_offset_px
        cv2.line(
            frame,
            tuple(np.round(ball_point).astype(int)),
            tuple(np.round(axis_point).astype(int)),
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        coordinate = centered_coordinate_cm(position_cm, center_cm)
        text_point = axis_point - normal * 12.0 + direction * 6.0
        draw_outlined_text(
            frame,
            f"x={coordinate:+.2f}",
            tuple(np.round(text_point).astype(int)),
            0.46,
            (0, 220, 255),
            2,
        )


def draw_overlay(
    frame: np.ndarray,
    geometry: TrackGeometry,
    candidate: BallCandidate | None,
    frame_box: np.ndarray | None,
    frame_center: tuple[float, float] | None,
    trail: deque[tuple[int, int]],
    position_cm: float | None,
    velocity_cm_s: float | None,
    predicted: bool,
    calibration: TrackCalibration | None,
    coordinate_config: dict,
    latency_ms: float,
    loop_fps: float,
    yolo_fps: float,
) -> None:
    cv2.polylines(
        frame,
        [np.round(geometry.source_points).astype(np.int32)],
        True,
        (140, 140, 140),
        1,
    )
    draw_coordinate_axis(
        frame, geometry, calibration, position_cm, coordinate_config
    )
    if frame_box is not None:
        x1, y1, x2, y2 = frame_box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    if frame_center is not None:
        cv2.circle(
            frame,
            (round(frame_center[0]), round(frame_center[1])),
            4,
            (0, 0, 255),
            -1,
        )
    for first, second in zip(trail, list(trail)[1:]):
        cv2.line(frame, first, second, (255, 180, 0), 2)

    state = "PRED" if predicted else "YOLO" if candidate is not None else "LOST"
    center_cm = float(coordinate_config.get("center_cm", 12.5))
    position_text = (
        "--"
        if position_cm is None
        else f"{centered_coordinate_cm(position_cm, center_cm):+.3f} cm"
    )
    velocity_text = "--" if velocity_cm_s is None else f"{velocity_cm_s:+.2f} cm/s"
    pixel_text = (
        "--"
        if frame_center is None
        else f"({frame_center[0]:.1f}, {frame_center[1]:.1f}) px"
    )
    lines = (
        f"{state}  x={position_text}  v={velocity_text}",
        f"center={pixel_text}",
        f"latency={latency_ms:.1f} ms  loop={loop_fps:.1f} fps  yolo={yolo_fps:.1f} fps",
    )
    for index, text in enumerate(lines):
        origin = (8, 22 + index * 22)
        cv2.putText(
            frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.51, (0, 0, 0), 3
        )
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.51,
            (255, 255, 255),
            1,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RKNN steel-ball vision tracker for Orange Pi 5B."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "vision.yaml"
    )
    parser.add_argument("--camera-device")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.camera_device:
        config["camera"]["device"] = args.camera_device
    model_path = project_path(args.model or config["model"])
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    calibration_path = project_path(config["calibration"])
    calibration = (
        TrackCalibration.load(calibration_path)
        if calibration_path.is_file()
        else None
    )
    if calibration is None:
        print(
            f"Track calibration not found: {calibration_path}; centimetre output is disabled.",
            file=sys.stderr,
        )
    geometry_path = project_path(config.get("geometry", "config/track_geometry.yaml"))
    geometry = TrackGeometry.from_config(config["roi"], geometry_path)
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
    trajectory_path = project_path(config["output"]["csv"])
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    display = bool(config["output"].get("display", True)) and not args.headless

    runtime = initialize_rknn(model_path)
    use_v3_center = uses_deployment_v3_center(config)
    worker_config = {**config["detection"], **config.get("yolo", {})}
    yolo_worker = (
        AsyncRoiYoloWorker(runtime, worker_config, config["roi"], geometry)
        if use_v3_center
        else AsyncYoloWorker(runtime, worker_config)
    )
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
    window_name = "Steel ball vision"

    try:
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

                if use_v3_center:
                    yolo_worker.submit(frame_id, timestamp_ns, frame, True)
                else:
                    rectified = stabilizer.apply(geometry.warp(frame))
                    yolo_worker.submit(frame_id, timestamp_ns, rectified, True)
                yolo_result = yolo_worker.latest()
                if (
                    yolo_result is not None
                    and yolo_result.completed_ns != last_yolo_completion_ns
                ):
                    last_yolo_completion_ns = yolo_result.completed_ns
                    yolo_completion_times.append(time.monotonic())
                    if yolo_result.error and yolo_result.error != last_yolo_error:
                        print(f"YOLO inference warning: {yolo_result.error}", file=sys.stderr)
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

                frame_center: tuple[float, float] | None = None
                frame_box: np.ndarray | None = None
                if candidate is not None:
                    frame_center = geometry.rectified_to_frame(
                        candidate.center_x, candidate.center_y
                    )
                    frame_box = geometry.box_to_frame(candidate.box)

                measurement = None
                if measurement_candidate is not None and calibration is not None:
                    measurement_x, measurement_y = geometry.rectified_to_frame(
                        measurement_candidate.center_x, measurement_candidate.center_y
                    )
                    measurement = calibration.position_cm(measurement_x, measurement_y)
                    if yolo_result is not None and tracker.state is not None:
                        age_seconds = max(
                            0.0, (timestamp_ns - yolo_result.timestamp_ns) / 1e9
                        )
                        measurement += float(tracker.state[1]) * age_seconds

                if calibration is not None:
                    estimate = tracker.step(
                        measurement,
                        timestamp_ns / 1e9,
                        measurement_std=yolo_measurement_std_cm,
                        innovation_gate_sigma=float(
                            config["fusion"]["innovation_gate_sigma"]
                        ),
                    )
                    position_cm = estimate.position_cm
                    velocity_cm_s = estimate.velocity_cm_s
                    detected = estimate.detected
                    predicted = estimate.predicted
                else:
                    position_cm = None
                    velocity_cm_s = None
                    detected = candidate is not None
                    predicted = False

                if frame_center is not None and detected:
                    trail.append(
                        (round(frame_center[0]), round(frame_center[1]))
                    )
                confidence = candidate.confidence if candidate is not None else None
                writer.writerow(
                    {
                        "timestamp_ns": timestamp_ns,
                        "frame_id": frame_id,
                        "detected": int(detected),
                        "confidence": "" if confidence is None else f"{confidence:.6f}",
                        "x_px": "" if frame_center is None else f"{frame_center[0]:.3f}",
                        "y_px": "" if frame_center is None else f"{frame_center[1]:.3f}",
                        "s_cm": "" if position_cm is None else f"{position_cm:.5f}",
                        "velocity_cm_s": (
                            "" if velocity_cm_s is None else f"{velocity_cm_s:.5f}"
                        ),
                        "predicted": int(predicted),
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
                        position_cm,
                        velocity_cm_s,
                        predicted,
                        calibration,
                        config.get("coordinate", {}),
                        latency_ms,
                        loop_fps,
                        yolo_fps,
                    )
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None:
            camera.close()
        yolo_worker.close()
        runtime.release()
        cv2.destroyAllWindows()

    interval_values = np.asarray(processing_intervals, dtype=np.float64)
    latency_values = np.asarray(latencies, dtype=np.float64)
    performance = {
        "processed_frames": len(latencies),
        "last_camera_frame_id": last_camera_frame,
        "last_processed_frame_id": last_processed_frame,
        "duration_seconds": time.monotonic() - start,
        "fusion_fps_median": (
            float(np.median(1.0 / interval_values)) if interval_values.size else 0.0
        ),
        "latency_ms_p95": (
            float(np.percentile(latency_values, 95)) if latency_values.size else None
        ),
        "rss_max_mb": max(rss_samples_mb) if rss_samples_mb else None,
        "rss_growth_mb": (
            rss_samples_mb[-1] - rss_samples_mb[0]
            if len(rss_samples_mb) >= 2
            else None
        ),
    }
    performance_path = trajectory_path.with_suffix(".performance.json")
    performance_path.write_text(json.dumps(performance, indent=2), encoding="utf-8")
    print(json.dumps(performance, indent=2))
    print(f"Trajectory CSV: {trajectory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
