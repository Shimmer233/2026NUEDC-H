from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .camera_controls import apply_saved_camera_controls


class LatestFrameCamera:
    """Capture continuously while retaining only the newest frame."""

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
        if (actual_width, actual_height) != (
            int(config["width"]),
            int(config["height"]),
        ):
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
            "install the supplied RKNNLite2 2.3.2 wheel on RK3588/aarch64"
        ) from error
    runtime = RKNNLite(verbose=False)
    if runtime.load_rknn(str(model_path)) != 0:
        raise RuntimeError(f"unable to load RKNN model: {model_path}")
    core_mask = getattr(RKNNLite, "NPU_CORE_0_1_2", RKNNLite.NPU_CORE_AUTO)
    if runtime.init_runtime(core_mask=core_mask) != 0:
        runtime.release()
        raise RuntimeError("RKNN runtime initialization failed")
    return runtime
