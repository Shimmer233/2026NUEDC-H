from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
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
from steel_ball.control import (  # noqa: E402
    ActuatorCalibration,
    BalanceController,
    BalanceOutput,
    ControlState,
    ControllerGains,
)
from steel_ball.hybrid_vision import (  # noqa: E402
    AsyncYoloWorker,
    BallCandidate,
    TrackGeometry,
    TrackStabilizer,
)
from steel_ball.pd42s1 import MotorSnapshot, MotorWorker, PD42S1Connection  # noqa: E402
from steel_ball.session_zero import validate_session_zero_marker  # noqa: E402
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

CONTROL_FIELDS = (
    "timestamp_ns",
    "frame_id",
    "requested_target_cm",
    "target_cm",
    "position_cm",
    "velocity_cm_s",
    "vision_source",
    "cv_confidence",
    "yolo_confidence",
    "theta_cmd_deg",
    "motor_target_count",
    "motor_actual_count",
    "motor_position_error",
    "motor_arrived",
    "motor_stalled",
    "control_state",
    "serial_ok",
)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


class LatestFrameCamera:
    """Keep only the newest camera frame so processing can never grow a queue."""

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


@dataclass
class TouchState:
    target_point: tuple[int, int] | None = None
    estop_requested: bool = False


class TouchUi:
    def __init__(self, frame_width: int) -> None:
        self.state = TouchState()
        self.estop_rect = (frame_width - 108, 8, frame_width - 8, 54)

    def mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x1, y1, x2, y2 = self.estop_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.state.estop_requested = True
        else:
            self.state.target_point = (x, y)

    def consume_target(self) -> tuple[int, int] | None:
        value = self.state.target_point
        self.state.target_point = None
        return value


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


def coordinate_tick_values(
    track_length_cm: float = 25.0, center_cm: float = 12.5, step_cm: float = 0.5
) -> tuple[tuple[float, float], ...]:
    if track_length_cm <= 0 or step_cm <= 0:
        raise ValueError("coordinate dimensions must be positive")
    count = round(track_length_cm / step_cm)
    if abs(count * step_cm - track_length_cm) > 1e-6:
        raise ValueError("coordinate step must divide the track length")
    return tuple(
        (index * step_cm, index * step_cm - center_cm)
        for index in range(count + 1)
    )


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
    minor_step_cm = float(config.get("minor_tick_cm", 0.5))
    axis_offset_px = float(config.get("axis_offset_px", 18.0))
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
    axis_color = (40, 40, 40)
    cv2.arrowedLine(
        frame,
        tuple(np.round(axis_start).astype(int)),
        tuple(np.round(axis_end + direction * 12.0).astype(int)),
        axis_color,
        2,
        cv2.LINE_AA,
        tipLength=0.025,
    )
    label_interval = int(config.get("label_interval_cm", 2))
    for absolute_cm, coordinate_cm in coordinate_tick_values(
        length_cm, center_cm, minor_step_cm
    ):
        point_value = calibration_marker(calibration, absolute_cm)
        if point_value is None:
            continue
        point = np.asarray(point_value, dtype=np.float64) + normal * axis_offset_px
        is_major = abs(coordinate_cm - round(coordinate_cm)) < 1e-6
        tick_length = 8.0 if is_major else 4.0
        first = tuple(np.round(point - normal * tick_length / 2.0).astype(int))
        second = tuple(np.round(point + normal * tick_length / 2.0).astype(int))
        cv2.line(frame, first, second, axis_color, 1, cv2.LINE_AA)
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
    if position_cm is not None:
        ball_value = calibration_marker(calibration, position_cm)
        if ball_value is not None:
            ball_point = np.asarray(ball_value, dtype=np.float64)
            axis_point = ball_point + normal * axis_offset_px
            cv2.line(
                frame,
                tuple(np.round(ball_point).astype(int)),
                tuple(np.round(axis_point).astype(int)),
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.drawMarker(
                frame,
                tuple(np.round(axis_point).astype(int)),
                (0, 220, 255),
                cv2.MARKER_TRIANGLE_UP,
                12,
                2,
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
    target_marker: tuple[int, int] | None,
    target_cm: float | None,
    calibration: TrackCalibration | None,
    coordinate_config: dict,
    control_state: ControlState,
    motor: MotorSnapshot | None,
    latency_ms: float,
    fps: float,
    yolo_fps: float,
    touch: TouchUi,
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
        cv2.circle(frame, (round(frame_center[0]), round(frame_center[1])), 4, (0, 0, 255), -1)
    for first, second in zip(trail, list(trail)[1:]):
        cv2.line(frame, first, second, (255, 180, 0), 2)
    if target_marker is not None:
        cv2.drawMarker(frame, target_marker, (0, 200, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
    detected_state = "PRED" if predicted else candidate.source.upper() if candidate else "LOST"
    center_cm = float(coordinate_config.get("center_cm", 12.5))
    position_text = (
        "--"
        if position_cm is None
        else f"{centered_coordinate_cm(position_cm, center_cm):+.3f} cm"
    )
    velocity_text = "--" if velocity_cm_s is None else f"{velocity_cm_s:+.2f} cm/s"
    target_text = (
        "--"
        if target_cm is None
        else f"{centered_coordinate_cm(target_cm, center_cm):+.2f} cm"
    )
    motor_text = "--"
    if motor is not None:
        motor_text = f"{motor.actual_count if motor.actual_count is not None else '--'} cnt"
    lines = (
        f"{detected_state}  x={position_text}  v={velocity_text}  target={target_text}",
        f"{control_state.value}  motor={motor_text}",
        f"latency={latency_ms:.1f} ms  loop={fps:.1f} fps  yolo={yolo_fps:.1f} fps",
    )
    for index, text in enumerate(lines):
        origin = (8, 22 + index * 22)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.51, (0, 0, 0), 3)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.51, (255, 255, 255), 1)
    x1, y1, x2, y2 = touch.estop_rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 210), -1)
    cv2.putText(frame, "STOP", (x1 + 17, y1 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if control_state == ControlState.ESTOP:
        cv2.putText(
            frame,
            "ESTOP: CUT MOTOR POWER",
            (70, frame.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3,
        )


def wait_for_motor(worker: MotorWorker, timeout: float = 2.0) -> MotorSnapshot:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = worker.snapshot()
        if snapshot.connected and snapshot.actual_count is not None:
            return snapshot
        if snapshot.fault:
            raise RuntimeError(f"motor initialization failed: {snapshot.fault}")
        time.sleep(0.02)
    raise RuntimeError("motor did not answer on /dev/ttyS0")


def make_controller(
    config: dict, calibration: ActuatorCalibration, target_cm: float
) -> BalanceController:
    control = config["control"]
    gains_config = control["gains"]
    gains = ControllerGains(
        kp_deg_per_cm=float(gains_config["kp_deg_per_cm"]),
        kv_deg_per_cm_s=float(gains_config["kv_deg_per_cm_s"]),
        ki_deg_per_cm_s=float(gains_config["ki_deg_per_cm_s"]),
        identified=bool(gains_config.get("identified", False)),
    )
    return BalanceController(
        calibration,
        gains,
        initial_target_cm=target_cm,
        target_min_cm=float(control["target_min_cm"]),
        target_max_cm=float(control["target_max_cm"]),
        target_slew_cm_s=float(control["target_slew_cm_s"]),
        max_angle_deg=float(control["max_angle_deg"]),
        integral_band_cm=float(control["integral_band_cm"]),
        integral_angle_limit_deg=float(control["integral_angle_limit_deg"]),
    )


def identification_angle(
    elapsed: float, amplitude_deg: float, pulse_seconds: float, rest_seconds: float
) -> float:
    phases = (
        (0.0, rest_seconds),
        (amplitude_deg, pulse_seconds),
        (0.0, rest_seconds),
        (-amplitude_deg, pulse_seconds),
        (0.0, rest_seconds),
    )
    cycle = sum(duration for _, duration in phases)
    phase_time = elapsed % cycle
    for angle, duration in phases:
        if phase_time < duration:
            return angle
        phase_time -= duration
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid steel-ball tracker and PD42S1 controller.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "board.yaml")
    parser.add_argument("--camera-device")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--detection-only", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument(
        "--control-mode",
        choices=("observe", "dry-run", "identify", "balance"),
        default="observe",
    )
    parser.add_argument("--target-cm", type=float, default=12.5)
    parser.add_argument("--arm", action="store_true", help="Allow balance mode to move the motor")
    parser.add_argument("--identify-angle-deg", type=float, default=0.15)
    parser.add_argument("--identify-pulse-seconds", type=float, default=0.25)
    parser.add_argument("--identify-rest-seconds", type=float, default=1.0)
    parser.add_argument("--identify-cycles", type=int, default=3)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.camera_device:
        config["camera"]["device"] = args.camera_device
    model_path = args.model.resolve() if args.model else project_path(config["model"])
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    track_calibration_path = project_path(config["calibration"])
    calibration = (
        TrackCalibration.load(track_calibration_path)
        if track_calibration_path.is_file()
        else None
    )
    if args.control_mode != "observe" and calibration is None:
        raise RuntimeError("dry-run/balance requires config/track_calibration.yaml")

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

    actuator: ActuatorCalibration | None = None
    controller: BalanceController | None = None
    zero_ready = False
    if args.control_mode != "observe":
        actuator_path = project_path(config["motor"]["calibration"])
        if not actuator_path.is_file():
            raise RuntimeError(f"actuator calibration is missing: {actuator_path}")
        actuator = ActuatorCalibration.load(actuator_path)
        zero_ready = actuator.persistence_verified
        controller = make_controller(config, actuator, args.target_cm)
        if args.control_mode in ("identify", "balance") and args.arm:
            if not actuator.mapping_ready:
                raise RuntimeError("motor movement is blocked until angle mapping is fitted")
            if args.control_mode == "balance" and not controller.gains.identified:
                raise RuntimeError("balance is blocked until plant gains are identified")
            if args.control_mode == "identify":
                if not 0 < args.identify_angle_deg <= float(config["control"]["max_angle_deg"]):
                    raise RuntimeError("identification angle exceeds the configured safe angle")
                actuator.angle_to_count(args.identify_angle_deg)
                actuator.angle_to_count(-args.identify_angle_deg)
            if not actuator.persistence_verified:
                motor_cfg = config["motor"]
                gate_connection = PD42S1Connection(
                    device=str(motor_cfg["device"]),
                    baudrate=int(motor_cfg["baudrate"]),
                    address=int(motor_cfg["address"]),
                    timeout_seconds=float(motor_cfg["timeout_seconds"]),
                    stale_disabled_status_physically_verified=bool(
                        motor_cfg.get(
                            "stale_disabled_status_physically_verified", False
                        )
                    ),
                )
                try:
                    current_count = gate_connection.read_position()
                    validate_session_zero_marker(
                        project_path(
                            motor_cfg.get(
                                "session_zero_marker",
                                "runs/board/motor_session_zero.json",
                            )
                        ),
                        address=actuator.address,
                        zero_count=actuator.zero_count,
                        tolerance_counts=actuator.zero_tolerance_counts,
                        min_count=actuator.min_count,
                        max_count=actuator.max_count,
                        current_count=current_count,
                    )
                finally:
                    gate_connection.close()
                zero_ready = True

    runtime = initialize_rknn(model_path)
    yolo_worker = AsyncYoloWorker(runtime, {**config["detection"], **config.get("yolo", {})})
    camera = LatestFrameCamera(config["camera"])
    motor_worker: MotorWorker | None = None
    if args.control_mode in ("identify", "balance") and args.arm:
        assert actuator is not None
        motor_cfg = config["motor"]
        connection = PD42S1Connection(
            device=str(motor_cfg["device"]),
            baudrate=int(motor_cfg["baudrate"]),
            address=int(motor_cfg["address"]),
            timeout_seconds=float(motor_cfg["timeout_seconds"]),
            stale_disabled_status_physically_verified=bool(
                motor_cfg.get("stale_disabled_status_physically_verified", False)
            ),
        )
        motor_worker = MotorWorker(
            connection,
            actuator.min_count,
            actuator.max_count,
            speed_rpm=int(motor_cfg["speed_rpm"]),
            acceleration=int(motor_cfg["acceleration"]),
            command_hz=float(motor_cfg["command_hz"]),
            status_hz=float(motor_cfg["status_hz"]),
        )
        motor_worker.start()
        initial_motor = wait_for_motor(motor_worker)
        if not actuator.min_count <= int(initial_motor.actual_count) <= actuator.max_count:
            motor_worker.emergency_stop("startup position is outside soft limits")
            raise RuntimeError("startup motor position is outside calibrated soft limits")
        motor_worker.set_target(actuator.zero_count)
        motor_worker.set_armed(True)
        zero_deadline = time.monotonic() + float(config["motor"]["shutdown_timeout_seconds"])
        while time.monotonic() < zero_deadline:
            snapshot = motor_worker.snapshot()
            if snapshot.actual_count is not None and abs(
                snapshot.actual_count - actuator.zero_count
            ) <= actuator.zero_tolerance_counts:
                break
            if snapshot.fault or snapshot.estop:
                raise RuntimeError(f"motor failed while moving to zero: {snapshot.fault}")
            time.sleep(0.02)
        else:
            motor_worker.emergency_stop("motor failed to reach zero before startup")
            raise RuntimeError("motor failed to reach zero before startup")

    trajectory_path = project_path(config["output"]["csv"])
    control_path = project_path(config["output"]["control_csv"])
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    control_path.parent.mkdir(parents=True, exist_ok=True)
    display = bool(config["output"].get("display", True)) and not args.headless
    window_name = "Steel ball controller"
    touch = TouchUi(int(config["camera"]["width"]))
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, touch.mouse)

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
    next_process_ns = 0
    next_display_time = 0.0
    fusion_period_ns = round(1e9 / float(config["runtime"]["fusion_hz"]))
    display_period = 1.0 / float(config["runtime"]["display_hz"])
    start = time.monotonic()
    identify_duration = (
        args.identify_cycles
        * (
            3.0 * args.identify_rest_seconds
            + 2.0 * args.identify_pulse_seconds
        )
        if args.control_mode == "identify"
        else 0.0
    )
    effective_duration = args.duration or identify_duration
    control_state = (
        ControlState.ZERO_UNVERIFIED
        if actuator is not None and not zero_ready
        else ControlState.READY
    )
    last_control_output: BalanceOutput | None = None
    try:
        with (
            trajectory_path.open("w", newline="", encoding="utf-8", buffering=1) as trajectory_stream,
            control_path.open("w", newline="", encoding="utf-8", buffering=1) as control_stream,
        ):
            trajectory_writer = csv.DictWriter(trajectory_stream, fieldnames=TRAJECTORY_FIELDS)
            control_writer = csv.DictWriter(control_stream, fieldnames=CONTROL_FIELDS)
            trajectory_writer.writeheader()
            control_writer.writeheader()
            while not effective_duration or time.monotonic() - start < effective_duration:
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
                yolo_worker.submit(frame_id, timestamp_ns, rectified, True)
                yolo_result = yolo_worker.latest()
                if yolo_result is not None and yolo_result.completed_ns != last_yolo_completion_ns:
                    last_yolo_completion_ns = yolo_result.completed_ns
                    yolo_completion_times.append(time.monotonic())
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
                    if yolo_result is not None:
                        age_seconds = max(0.0, (timestamp_ns - yolo_result.timestamp_ns) / 1e9)
                        if tracker.state is not None:
                            measurement += float(tracker.state[1]) * age_seconds
                estimate = tracker.step(
                    measurement,
                    timestamp_ns / 1e9,
                    measurement_std=yolo_measurement_std_cm,
                    innovation_gate_sigma=float(config["fusion"]["innovation_gate_sigma"]),
                )
                if frame_center is not None and estimate.detected:
                    trail.append((round(frame_center[0]), round(frame_center[1])))

                motor_snapshot = motor_worker.snapshot() if motor_worker is not None else None
                if motor_snapshot is not None and motor_snapshot.estop:
                    control_state = ControlState.ESTOP
                elif motor_snapshot is not None and motor_snapshot.fault:
                    control_state = ControlState.MOTOR_FAULT
                elif not estimate.available:
                    control_state = ControlState.LOST
                elif motor_worker is not None:
                    control_state = ControlState.ARMED
                elif actuator is not None and not zero_ready:
                    control_state = ControlState.ZERO_UNVERIFIED
                else:
                    control_state = ControlState.READY

                if controller is not None and args.control_mode == "identify":
                    assert actuator is not None
                    angle = (
                        identification_angle(
                            time.monotonic() - start,
                            args.identify_angle_deg,
                            args.identify_pulse_seconds,
                            args.identify_rest_seconds,
                        )
                        if estimate.available
                        else 0.0
                    )
                    last_control_output = BalanceOutput(
                        target_cm=args.target_cm,
                        requested_target_cm=args.target_cm,
                        error_cm=(
                            None
                            if estimate.position_cm is None
                            else args.target_cm - estimate.position_cm
                        ),
                        angle_deg=angle,
                        target_count=actuator.angle_to_count(angle),
                        saturated=False,
                        state=(
                            ControlState.ARMED
                            if estimate.available
                            else ControlState.LOST
                        ),
                    )
                    if motor_worker is not None:
                        motor_worker.set_target(last_control_output.target_count)
                elif controller is not None:
                    target_click = touch.consume_target()
                    if target_click is not None and calibration is not None:
                        if cv2.pointPolygonTest(
                            geometry.source_points.astype(np.float32), target_click, False
                        ) >= 0:
                            controller.set_target(
                                calibration.position_cm(target_click[0], target_click[1])
                            )
                    last_control_output = controller.update(
                        estimate.position_cm,
                        estimate.velocity_cm_s,
                        timestamp_ns / 1e9,
                        estimate.available,
                        estimate.predicted,
                        motor_worker is not None and control_state == ControlState.ARMED,
                    )
                    if motor_worker is not None and not motor_snapshot.estop:
                        motor_worker.set_target(last_control_output.target_count)
                if touch.state.estop_requested:
                    control_state = ControlState.ESTOP
                    if motor_worker is not None:
                        motor_worker.emergency_stop("touchscreen emergency stop")

                confidence = candidate.confidence if candidate is not None else None
                trajectory_writer.writerow(
                    {
                        "timestamp_ns": timestamp_ns,
                        "frame_id": frame_id,
                        "detected": int(
                            estimate.detected if calibration is not None else candidate is not None
                        ),
                        "confidence": "" if confidence is None else f"{confidence:.6f}",
                        "x_px": "" if frame_center is None else f"{frame_center[0]:.3f}",
                        "y_px": "" if frame_center is None else f"{frame_center[1]:.3f}",
                        "s_cm": "" if estimate.position_cm is None else f"{estimate.position_cm:.5f}",
                        "velocity_cm_s": "" if estimate.velocity_cm_s is None else f"{estimate.velocity_cm_s:.5f}",
                        "predicted": int(estimate.predicted),
                    }
                )
                status = motor_snapshot.status if motor_snapshot is not None else None
                control_writer.writerow(
                    {
                        "timestamp_ns": timestamp_ns,
                        "frame_id": frame_id,
                        "requested_target_cm": "" if last_control_output is None else f"{last_control_output.requested_target_cm:.5f}",
                        "target_cm": "" if last_control_output is None else f"{last_control_output.target_cm:.5f}",
                        "position_cm": "" if estimate.position_cm is None else f"{estimate.position_cm:.5f}",
                        "velocity_cm_s": "" if estimate.velocity_cm_s is None else f"{estimate.velocity_cm_s:.5f}",
                        "vision_source": (
                            "yolo"
                            if estimate.detected
                            else "kalman"
                            if estimate.predicted
                            else ""
                        ),
                        "cv_confidence": "",
                        "yolo_confidence": "" if yolo_result is None or yolo_result.candidate is None else f"{yolo_result.candidate.confidence:.6f}",
                        "theta_cmd_deg": "" if last_control_output is None else f"{last_control_output.angle_deg:.6f}",
                        "motor_target_count": "" if motor_snapshot is None or motor_snapshot.target_count is None else motor_snapshot.target_count,
                        "motor_actual_count": "" if motor_snapshot is None or motor_snapshot.actual_count is None else motor_snapshot.actual_count,
                        "motor_position_error": "" if status is None else status.position_error,
                        "motor_arrived": "" if status is None else int(status.arrived),
                        "motor_stalled": "" if status is None else int(status.stalled),
                        "control_state": control_state.value,
                        "serial_ok": int(motor_snapshot is not None and motor_snapshot.connected and not motor_snapshot.fault),
                    }
                )

                latency_ms = (time.perf_counter() - processing_start) * 1000.0
                latencies.append(latency_ms)
                processed_at = time.monotonic()
                if previous_processed_at is not None:
                    processing_intervals.append(processed_at - previous_processed_at)
                previous_processed_at = processed_at
                frame_times.append(processed_at)
                if resource is not None and (not rss_samples_mb or len(latencies) % 240 == 0):
                    rss_samples_mb.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
                fps = (
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
                    target_cm = (
                        last_control_output.target_cm if last_control_output is not None else None
                    )
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
                        calibration_marker(calibration, target_cm),
                        target_cm,
                        calibration,
                        config.get("coordinate", {}),
                        control_state,
                        motor_snapshot,
                        latency_ms,
                        fps,
                        yolo_fps,
                        touch,
                    )
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key in (ord("e"), ord("E")):
                        touch.state.estop_requested = True
                if control_state == ControlState.ESTOP:
                    print("ESTOP latched: use the physical motor power switch now.", file=sys.stderr)
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        yolo_worker.close()
        if motor_worker is not None:
            snapshot = motor_worker.snapshot()
            if not snapshot.estop:
                returned = motor_worker.return_zero_and_disable(
                    actuator.zero_count if actuator is not None else 0,
                    actuator.zero_tolerance_counts if actuator is not None else 20,
                    float(config["motor"]["shutdown_timeout_seconds"]),
                )
                if not returned:
                    motor_worker.emergency_stop("failed to return to zero during shutdown")
                    print("Motor did not return to zero; use the physical power switch.", file=sys.stderr)
            motor_worker.close()
        runtime.release()
        cv2.destroyAllWindows()

    interval_values = np.asarray(processing_intervals, dtype=np.float64)
    latency_values = np.asarray(latencies, dtype=np.float64)
    performance = {
        "processed_frames": len(latencies),
        "last_camera_frame_id": last_camera_frame,
        "last_processed_frame_id": last_processed_frame,
        "duration_seconds": time.monotonic() - start,
        "fusion_fps_median": float(np.median(1.0 / interval_values)) if interval_values.size else 0.0,
        "latency_ms_p95": float(np.percentile(latency_values, 95)) if latency_values.size else None,
        "rss_max_mb": max(rss_samples_mb) if rss_samples_mb else None,
        "rss_growth_mb": rss_samples_mb[-1] - rss_samples_mb[0] if len(rss_samples_mb) >= 2 else None,
        "passes_80_fps": bool(interval_values.size and np.median(1.0 / interval_values) >= 80.0),
    }
    performance_path = trajectory_path.with_suffix(".performance.json")
    performance_path.write_text(json.dumps(performance, indent=2), encoding="utf-8")
    print(json.dumps(performance, indent=2))
    print(f"Trajectory CSV: {trajectory_path}")
    print(f"Control CSV: {control_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
