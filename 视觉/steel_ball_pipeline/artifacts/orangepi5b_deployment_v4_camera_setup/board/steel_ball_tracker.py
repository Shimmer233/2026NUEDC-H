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
except ImportError:  # Windows development host; the board runs Linux.
    resource = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.calibration import TrackCalibration  # noqa: E402
from steel_ball.camera_controls import apply_saved_camera_controls  # noqa: E402
from steel_ball.rknn_postprocess import decode_yolov5_heads  # noqa: E402
from steel_ball.tracking import PositionVelocityKalman  # noqa: E402
from steel_ball.vision import (  # noqa: E402
    letterbox_black,
    refine_ball_center,
    rknn_nhwc_batch,
    unletterbox_box,
)


CSV_FIELDS = (
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


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


class LatestFrameCamera:
    """Keep only the newest camera frame so inference can never grow a queue."""

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
        if (actual_width, actual_height) != (int(config["width"]), int(config["height"])):
            self.capture.release()
            raise RuntimeError(
                f"camera rejected {config['width']}x{config['height']}; "
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
        self.thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
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

    def next(self, after_id: int, timeout: float = 1.0) -> tuple[int, int, np.ndarray] | None:
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
        raise RuntimeError("install Rockchip rknn_toolkit_lite2 2.3.2 for RK3588/aarch64") from error
    runtime = RKNNLite(verbose=False)
    if runtime.load_rknn(str(model_path)) != 0:
        raise RuntimeError(f"unable to load RKNN model: {model_path}")
    core_mask = getattr(RKNNLite, "NPU_CORE_0_1_2", RKNNLite.NPU_CORE_AUTO)
    if runtime.init_runtime(core_mask=core_mask) != 0:
        runtime.release()
        raise RuntimeError("RKNN runtime initialization failed")
    return runtime


def draw_overlay(
    frame: np.ndarray,
    roi: dict,
    box: np.ndarray | None,
    center: tuple[float, float] | None,
    trail: deque[tuple[int, int]],
    position: float | None,
    velocity: float | None,
    predicted: bool,
    latency_ms: float,
    fps: float,
) -> None:
    x, y, width, height = (int(roi[key]) for key in ("x", "y", "width", "height"))
    cv2.rectangle(frame, (x, y), (x + width, y + height), (120, 120, 120), 1)
    if box is not None:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    if center is not None:
        cv2.circle(frame, (round(center[0]), round(center[1])), 3, (0, 0, 255), -1)
    for first, second in zip(trail, list(trail)[1:]):
        cv2.line(frame, first, second, (255, 180, 0), 2)
    state = "PRED" if predicted else "DETECT" if center is not None else "LOST"
    position_text = "--" if position is None else f"{position:.3f} cm"
    velocity_text = "--" if velocity is None else f"{velocity:+.2f} cm/s"
    lines = (
        f"{state}  s={position_text}  v={velocity_text}",
        f"latency={latency_ms:.1f} ms  fps={fps:.1f}",
    )
    for index, text in enumerate(lines):
        origin = (8, 24 + index * 24)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time RK3588 steel-ball detector and tracker.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "board.yaml")
    parser.add_argument("--camera-device", help="Override the V4L2 camera, for example /dev/video2")
    parser.add_argument("--model", type=Path, help="Override the RKNN model path")
    parser.add_argument("--detection-only", action="store_true", help="Run boxes and centres without cm calibration")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds; zero runs forever.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.camera_device:
        config["camera"]["device"] = args.camera_device
    model_path = args.model.resolve() if args.model else project_path(config["model"])
    calibration = (
        None
        if args.detection_only
        else TrackCalibration.load(project_path(config["calibration"]))
    )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    runtime = initialize_rknn(model_path)
    camera = LatestFrameCamera(config["camera"])
    tracker = PositionVelocityKalman(
        process_acceleration_std=float(config["tracking"]["process_acceleration_std"]),
        measurement_std=float(config["tracking"]["measurement_std_cm"]),
        max_prediction_frames=int(config["tracking"]["max_prediction_frames"]),
    )
    csv_path = project_path(config["output"]["csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    display = bool(config["output"].get("display", True)) and not args.headless
    roi = config["roi"]
    trail: deque[tuple[int, int]] = deque(maxlen=120)
    frame_times: deque[float] = deque(maxlen=120)
    processing_intervals: list[float] = []
    latencies: list[float] = []
    rss_samples_mb: list[float] = []
    previous_processed_at: float | None = None
    skipped_camera_frames = 0
    last_frame_id = -1
    start = time.monotonic()
    try:
        with csv_path.open("w", newline="", encoding="utf-8", buffering=1) as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            while not args.duration or time.monotonic() - start < args.duration:
                packet = camera.next(last_frame_id)
                if packet is None:
                    continue
                frame_id, timestamp_ns, frame = packet
                if last_frame_id >= 0:
                    skipped_camera_frames += max(0, frame_id - last_frame_id - 1)
                last_frame_id = frame_id
                processing_start = time.perf_counter()
                x, y, width, height = (int(roi[key]) for key in ("x", "y", "width", "height"))
                crop = frame[y : y + height, x : x + width]
                if crop.shape[:2] != (height, width):
                    raise RuntimeError(f"ROI {roi} exceeds camera frame shape {frame.shape}")
                roi_input_width = int(roi.get("output_width", width))
                roi_input_height = int(roi.get("output_height", height))
                if (roi_input_width, roi_input_height) != (width, height):
                    detection_crop = cv2.resize(
                        crop,
                        (roi_input_width, roi_input_height),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    detection_crop = crop
                model_input, letterbox_meta = letterbox_black(
                    detection_crop, int(config["detection"]["input_size"])
                )
                rgb = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
                outputs = runtime.inference(
                    inputs=[rknn_nhwc_batch(rgb)], data_format=["nhwc"]
                )
                if outputs is None:
                    raise RuntimeError("RKNN inference returned no outputs")
                detections = decode_yolov5_heads(
                    outputs,
                    input_size=int(config["detection"]["input_size"]),
                    confidence_threshold=float(config["detection"]["confidence_threshold"]),
                    iou_threshold=float(config["detection"]["iou_threshold"]),
                )
                detection = None
                selected_box_roi = None
                track_y_min = float(config["detection"].get("track_y_min", 0))
                track_y_max = float(config["detection"].get("track_y_max", height))
                for candidate in detections:
                    candidate_box = unletterbox_box(candidate.box, letterbox_meta)
                    center_y = float((candidate_box[1] + candidate_box[3]) / 2)
                    if track_y_min <= center_y <= track_y_max:
                        detection = candidate
                        scale = np.asarray(
                            [
                                width / roi_input_width,
                                height / roi_input_height,
                                width / roi_input_width,
                                height / roi_input_height,
                            ],
                            dtype=np.float32,
                        )
                        selected_box_roi = candidate_box * scale
                        break
                box_global: np.ndarray | None = None
                center: tuple[float, float] | None = None
                confidence: float | None = None
                measurement: float | None = None
                if detection is not None:
                    assert selected_box_roi is not None
                    box_global = selected_box_roi + np.asarray([x, y, x, y], dtype=np.float32)
                    center_x, center_y, _ = refine_ball_center(frame, box_global)
                    center = (center_x, center_y)
                    confidence = detection.confidence
                    measurement = (
                        calibration.position_cm(center_x, center_y)
                        if calibration is not None
                        else None
                    )
                    trail.append((round(center_x), round(center_y)))
                estimate = tracker.step(measurement, timestamp_ns / 1e9)
                latency_ms = (time.perf_counter() - processing_start) * 1000.0
                latencies.append(latency_ms)
                processed_at = time.monotonic()
                if previous_processed_at is not None:
                    processing_intervals.append(processed_at - previous_processed_at)
                previous_processed_at = processed_at
                frame_times.append(processed_at)
                if resource is not None and (not rss_samples_mb or len(latencies) % 60 == 0):
                    rss_samples_mb.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
                if len(frame_times) >= 2:
                    fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                else:
                    fps = 0.0
                writer.writerow(
                    {
                        "timestamp_ns": timestamp_ns,
                        "frame_id": frame_id,
                        "detected": int(detection is not None),
                        "confidence": "" if confidence is None else f"{confidence:.6f}",
                        "x_px": "" if center is None else f"{center[0]:.3f}",
                        "y_px": "" if center is None else f"{center[1]:.3f}",
                        "s_cm": "" if estimate.position_cm is None else f"{estimate.position_cm:.5f}",
                        "velocity_cm_s": "" if estimate.velocity_cm_s is None else f"{estimate.velocity_cm_s:.5f}",
                        "predicted": int(estimate.predicted),
                    }
                )
                if display:
                    draw_overlay(
                        frame,
                        roi,
                        box_global,
                        center,
                        trail,
                        estimate.position_cm,
                        estimate.velocity_cm_s,
                        estimate.predicted,
                        latency_ms,
                        fps,
                    )
                    cv2.imshow("Steel ball tracker", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        runtime.release()
        cv2.destroyAllWindows()
    interval_values = np.asarray(processing_intervals, dtype=np.float64)
    latency_values = np.asarray(latencies, dtype=np.float64)
    performance = {
        "processed_frames": len(latencies),
        "skipped_camera_frames": skipped_camera_frames,
        "duration_seconds": time.monotonic() - start,
        "processing_fps_median": (
            float(np.median(1.0 / interval_values)) if interval_values.size else 0.0
        ),
        "latency_ms_p95": (
            float(np.percentile(latency_values, 95)) if latency_values.size else None
        ),
        "rss_max_mb": max(rss_samples_mb) if rss_samples_mb else None,
        "rss_growth_mb": (
            rss_samples_mb[-1] - rss_samples_mb[0] if len(rss_samples_mb) >= 2 else None
        ),
        "passes_58_fps": bool(
            interval_values.size and np.median(1.0 / interval_values) >= 58.0
        ),
        "passes_p95_50ms": bool(
            latency_values.size and np.percentile(latency_values, 95) <= 50.0
        ),
    }
    csv_path.with_suffix(".performance.json").write_text(
        json.dumps(performance, indent=2), encoding="utf-8"
    )
    print(json.dumps(performance, indent=2))
    print(f"Trajectory CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
