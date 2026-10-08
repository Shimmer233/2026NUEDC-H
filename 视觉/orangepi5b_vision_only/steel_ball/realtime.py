from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .calibration import TrackCalibration
from .camera_controls import apply_saved_camera_controls
from .detector import (
    AsyncRoiYoloWorker,
    AsyncYoloWorker,
    TrackGeometry,
    TrackStabilizer,
)
from .tracking import PositionVelocityKalman


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def uses_deployment_v3_center(config: dict) -> bool:
    mode = str(config.get("center_detection", {}).get("mode", "rectified")).lower()
    return mode in {"deployment_v3_roi", "direct_roi", "roi"}


@dataclass(frozen=True)
class BallState:
    """One immutable, thread-safe position and velocity reading."""

    timestamp_ns: int
    frame_id: int
    valid: bool
    detected: bool
    predicted: bool
    confidence: float | None
    x_px: float | None
    y_px: float | None
    position_cm: float | None
    centered_position_cm: float | None
    velocity_cm_s: float | None
    latency_ms: float
    inference_error: str | None = None
    frame: np.ndarray | None = None

    @property
    def x_cm(self) -> float | None:
        """Position relative to the track center; left is negative."""
        return self.centered_position_cm

    @property
    def v_cm_s(self) -> float | None:
        """Velocity along the track; motion to the right is positive."""
        return self.velocity_cm_s


class LatestFrameCamera:
    """Read V4L2 continuously while retaining only the newest frame."""

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
            apply_saved_camera_controls(control_device, config.get("controls", {}))
        except RuntimeError as error:
            print(f"Camera control warning: {error}", file=sys.stderr)

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
        self, after_id: int, timeout: float = 0.2
    ) -> tuple[int, int, np.ndarray] | None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.stopped and self.frame_id <= after_id:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            if self.frame is None or self.frame_id <= after_id:
                return None
            return self.frame_id, self.timestamp_ns, self.frame.copy()

    def close(self) -> None:
        if self.stopped:
            return
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


class SteelBallVision:
    """Background RKNN tracker exposing the latest ball state as Python data."""

    def __init__(
        self,
        config_path: str | Path = PROJECT_ROOT / "config" / "vision.yaml",
        camera_device: str | None = None,
        model_path: str | Path | None = None,
        *,
        runtime: object | None = None,
        camera: object | None = None,
        publish_frame: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if camera_device is not None:
            self.config["camera"]["device"] = camera_device
        self.model_path = project_path(model_path or self.config["model"])
        if runtime is None and not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)

        calibration_path = project_path(self.config["calibration"])
        if not calibration_path.is_file():
            raise RuntimeError(
                f"track calibration is required for position output: {calibration_path}"
            )
        self.calibration = TrackCalibration.load(calibration_path)
        geometry_path = project_path(
            self.config.get("geometry", "config/track_geometry.yaml")
        )
        self.geometry = TrackGeometry.from_config(self.config["roi"], geometry_path)
        self.stabilizer = TrackStabilizer(
            self.config.get("stabilization", {}),
            (self.geometry.output_width, self.geometry.output_height),
        )
        self.use_v3_center = uses_deployment_v3_center(self.config)
        self.tracker = PositionVelocityKalman(
            process_acceleration_std=float(
                self.config["tracking"]["process_acceleration_std"]
            ),
            measurement_std=float(self.config["tracking"]["measurement_std_cm"]),
            max_prediction_frames=int(
                self.config["tracking"]["max_prediction_frames"]
            ),
        )
        self.center_cm = float(self.config.get("coordinate", {}).get("center_cm", 12.5))

        self._provided_runtime = runtime
        self._provided_camera = camera
        self.publish_frame = publish_frame
        self._runtime: object | None = None
        self._camera: object | None = None
        self._worker: AsyncYoloWorker | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest: BallState | None = None
        self._fatal_error: BaseException | None = None
        self._started = False
        self._closed = False

    @property
    def latest(self) -> BallState | None:
        """Return the newest state immediately, or None before the first update."""
        with self._condition:
            return self._latest

    @property
    def error(self) -> BaseException | None:
        with self._condition:
            return self._fatal_error

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> "SteelBallVision":
        if self._closed:
            raise RuntimeError("SteelBallVision cannot restart after close()")
        if self.running:
            return self
        if self._started:
            raise RuntimeError("SteelBallVision background thread has already stopped")
        self._started = True
        try:
            self._runtime = (
                self._provided_runtime
                if self._provided_runtime is not None
                else initialize_rknn(self.model_path)
            )
            worker_config = {**self.config["detection"], **self.config.get("yolo", {})}
            self._worker = (
                AsyncRoiYoloWorker(
                    self._runtime,
                    worker_config,
                    self.config["roi"],
                    self.geometry,
                )
                if self.use_v3_center
                else AsyncYoloWorker(self._runtime, worker_config)
            )
            self._camera = (
                self._provided_camera
                if self._provided_camera is not None
                else LatestFrameCamera(self.config["camera"])
            )
            self._thread = threading.Thread(
                target=self._run, name="steel-ball-vision", daemon=True
            )
            self._thread.start()
            return self
        except BaseException:
            self._release_resources()
            self._closed = True
            raise

    def wait_for_update(
        self, after_frame_id: int = -1, timeout: float | None = None
    ) -> BallState | None:
        """Wait until a state newer than after_frame_id is available."""
        if not self._started:
            raise RuntimeError("call start() or use 'with SteelBallVision(...)' first")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if self._latest is not None and self._latest.frame_id > after_frame_id:
                    return self._latest
                if self._fatal_error is not None:
                    raise RuntimeError("steel-ball vision background thread failed") from self._fatal_error
                if self._closed or (self._started and not self.running):
                    return None
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if remaining == 0.0:
                    return None
                self._condition.wait(remaining)

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        camera = self._camera
        if camera is not None:
            camera.close()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._release_resources(close_camera=False)
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def __enter__(self) -> "SteelBallVision":
        return self.start()

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def _release_resources(self, close_camera: bool = True) -> None:
        if close_camera and self._camera is not None:
            self._camera.close()
        self._camera = None
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        if self._runtime is not None:
            release = getattr(self._runtime, "release", None)
            if callable(release):
                release()
            self._runtime = None

    def _publish(self, state: BallState) -> None:
        with self._condition:
            self._latest = state
            self._condition.notify_all()

    def _run(self) -> None:
        assert self._camera is not None and self._worker is not None
        last_camera_frame = -1
        last_yolo_result_id = -1
        next_process_ns = 0
        fusion_period_ns = round(1e9 / float(self.config["runtime"]["fusion_hz"]))
        yolo_max_age_ms = float(self.config["fusion"].get("yolo_max_age_ms", 80.0))
        measurement_std = float(
            self.config["fusion"].get("yolo_measurement_std_cm", 0.12)
        )
        innovation_gate = float(self.config["fusion"].get("innovation_gate_sigma", 5.0))
        try:
            while not self._stop.is_set():
                packet = self._camera.next(last_camera_frame, timeout=0.2)
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
                processing_start = time.perf_counter()

                if self.use_v3_center:
                    self._worker.submit(frame_id, timestamp_ns, frame, True)
                else:
                    rectified = self.stabilizer.apply(self.geometry.warp(frame))
                    self._worker.submit(frame_id, timestamp_ns, rectified, True)
                result = self._worker.latest()
                is_new = result is not None and result.frame_id != last_yolo_result_id
                age_ms = (
                    float("inf")
                    if result is None
                    else max(0.0, (timestamp_ns - result.timestamp_ns) / 1e6)
                )
                candidate = (
                    result.candidate
                    if result is not None
                    and result.error is None
                    and age_ms <= yolo_max_age_ms
                    else None
                )
                measurement_candidate = candidate if is_new else None
                if is_new and result is not None:
                    last_yolo_result_id = result.frame_id

                frame_x: float | None = None
                frame_y: float | None = None
                if candidate is not None:
                    frame_x, frame_y = self.geometry.rectified_to_frame(
                        candidate.center_x, candidate.center_y
                    )

                measurement = None
                if measurement_candidate is not None:
                    measurement_x, measurement_y = self.geometry.rectified_to_frame(
                        measurement_candidate.center_x,
                        measurement_candidate.center_y,
                    )
                    measurement = self.calibration.position_cm(
                        measurement_x, measurement_y
                    )
                    if result is not None and self.tracker.state is not None:
                        measurement_age_s = max(
                            0.0, (timestamp_ns - result.timestamp_ns) / 1e9
                        )
                        measurement += float(self.tracker.state[1]) * measurement_age_s

                estimate = self.tracker.step(
                    measurement,
                    timestamp_ns / 1e9,
                    measurement_std=measurement_std,
                    innovation_gate_sigma=innovation_gate,
                )
                position_cm = estimate.position_cm
                centered_position_cm = (
                    None if position_cm is None else position_cm - self.center_cm
                )
                self._publish(
                    BallState(
                        timestamp_ns=timestamp_ns,
                        frame_id=frame_id,
                        valid=estimate.available,
                        detected=estimate.detected,
                        predicted=estimate.predicted,
                        confidence=(
                            None if candidate is None else candidate.confidence
                        ),
                        x_px=frame_x,
                        y_px=frame_y,
                        position_cm=position_cm,
                        centered_position_cm=centered_position_cm,
                        velocity_cm_s=estimate.velocity_cm_s,
                        latency_ms=(time.perf_counter() - processing_start) * 1000.0,
                        inference_error=None if result is None else result.error,
                        frame=frame.copy() if self.publish_frame else None,
                    )
                )
        except BaseException as error:
            with self._condition:
                self._fatal_error = error
                self._condition.notify_all()
        finally:
            with self._condition:
                self._condition.notify_all()
