from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .calibration import TrackCalibration
from .rknn_postprocess import decode_yolov5_heads
from .vision import (
    letterbox_black,
    refine_ball_center,
    rknn_nhwc_batch,
    unletterbox_box,
)


@dataclass(frozen=True)
class BallCandidate:
    center_x: float
    center_y: float
    box: np.ndarray
    confidence: float
    source: str
    radius: float


@dataclass(frozen=True)
class YoloResult:
    frame_id: int
    timestamp_ns: int
    completed_ns: int
    candidate: BallCandidate | None
    error: str | None = None


class TrackGeometry:
    def __init__(self, source_points: np.ndarray, output_size: tuple[int, int]) -> None:
        source = np.asarray(source_points, dtype=np.float32)
        if source.shape != (4, 2):
            raise ValueError("track geometry requires four source points")
        width, height = output_size
        if width <= 0 or height <= 0:
            raise ValueError("track geometry output size is invalid")
        destination = np.asarray(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        self.source_points = source
        self.output_width = int(width)
        self.output_height = int(height)
        self.matrix = cv2.getPerspectiveTransform(source, destination)
        self.inverse_matrix = cv2.getPerspectiveTransform(destination, source)

    @classmethod
    def from_config(cls, roi: dict, geometry_path: Path | None = None) -> "TrackGeometry":
        if geometry_path is not None and geometry_path.is_file():
            payload = yaml.safe_load(geometry_path.read_text(encoding="utf-8"))
            return cls(
                np.asarray(payload["source_points"], dtype=np.float32),
                (int(payload["output_width"]), int(payload["output_height"])),
            )
        x, y, width, height = (
            float(roi[key]) for key in ("x", "y", "width", "height")
        )
        source = np.asarray(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            dtype=np.float32,
        )
        return cls(
            source,
            (
                int(roi.get("output_width", width)),
                int(roi.get("output_height", height)),
            ),
        )

    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(
            frame,
            self.matrix,
            (self.output_width, self.output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def rectified_to_frame(self, x: float, y: float) -> tuple[float, float]:
        point = np.asarray([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.inverse_matrix)[0, 0]
        return float(mapped[0]), float(mapped[1])

    def frame_to_rectified(self, x: float, y: float) -> tuple[float, float]:
        point = np.asarray([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.matrix)[0, 0]
        return float(mapped[0]), float(mapped[1])

    def box_to_frame(self, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
        points = np.asarray(
            [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32
        )
        mapped = cv2.perspectiveTransform(points, self.inverse_matrix)[0]
        return np.asarray(
            [
                mapped[:, 0].min(),
                mapped[:, 1].min(),
                mapped[:, 0].max(),
                mapped[:, 1].max(),
            ],
            dtype=np.float32,
        )

    def position_cm(
        self, candidate: BallCandidate, calibration: TrackCalibration
    ) -> tuple[float, float, float]:
        frame_x, frame_y = self.rectified_to_frame(
            candidate.center_x, candidate.center_y
        )
        return calibration.position_cm(frame_x, frame_y), frame_x, frame_y


class TrackStabilizer:
    """Remove small camera translations in the rectified measurement ROI."""

    def __init__(self, config: dict, output_size: tuple[int, int]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.max_shift_px = float(config.get("max_shift_px", 8.0))
        self.min_response = float(config.get("min_response", 0.08))
        width, height = output_size
        self.window = cv2.createHanningWindow((int(width), int(height)), cv2.CV_32F)
        self.reference: np.ndarray | None = None
        self.last_shift = (0.0, 0.0)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.reference is None:
            self.reference = gray
            return image
        (shift_x, shift_y), response = cv2.phaseCorrelate(
            self.reference, gray, self.window
        )
        if (
            not np.isfinite(shift_x)
            or not np.isfinite(shift_y)
            or response < self.min_response
            or abs(shift_x) > self.max_shift_px
            or abs(shift_y) > self.max_shift_px
        ):
            self.last_shift = (0.0, 0.0)
            return image
        self.last_shift = (float(shift_x), float(shift_y))
        transform = np.asarray(
            [[1.0, 0.0, -shift_x], [0.0, 1.0, -shift_y]], dtype=np.float32
        )
        return cv2.warpAffine(
            image,
            transform,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )


class AsyncYoloWorker:
    """Run RKNN inference asynchronously and retain only the newest request."""

    def __init__(self, runtime: object, config: dict) -> None:
        self.runtime = runtime
        self.input_size = int(config.get("input_size", 640))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.03))
        self.iou_threshold = float(config.get("iou_threshold", 0.45))
        self.track_y_min = float(config.get("track_y_min", 0))
        self.track_y_max = float(config.get("track_y_max", 110))
        self.normal_hz = float(config.get("normal_hz", 50.0))
        self.reacquire_hz = float(config.get("reacquire_hz", 50.0))
        self._condition = threading.Condition()
        self._request: tuple[int, int, np.ndarray, bool] | None = None
        self._result: YoloResult | None = None
        self._stop = False
        self._thread = threading.Thread(
            target=self._run, name="rknn-yolo", daemon=True
        )
        self._thread.start()

    def submit(
        self, frame_id: int, timestamp_ns: int, rectified: np.ndarray, urgent: bool
    ) -> None:
        with self._condition:
            self._request = (frame_id, timestamp_ns, rectified.copy(), urgent)
            self._condition.notify_all()

    def latest(self) -> YoloResult | None:
        with self._condition:
            return self._result

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _infer(self, rectified: np.ndarray) -> BallCandidate | None:
        model_input, meta = letterbox_black(rectified, self.input_size)
        rgb = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
        outputs = self.runtime.inference(
            inputs=[rknn_nhwc_batch(rgb)], data_format=["nhwc"]
        )
        if outputs is None:
            raise RuntimeError("RKNN inference returned no outputs")
        detections = decode_yolov5_heads(
            outputs,
            input_size=self.input_size,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )
        for detection in detections:
            box = unletterbox_box(detection.box, meta)
            center_y = float((box[1] + box[3]) / 2.0)
            if not self.track_y_min <= center_y <= self.track_y_max:
                continue
            center_x, center_y, _ = refine_ball_center(rectified, box)
            radius = max(2.0, 0.25 * ((box[2] - box[0]) + (box[3] - box[1])))
            return BallCandidate(
                center_x,
                center_y,
                box.astype(np.float32),
                detection.confidence,
                "yolo",
                float(radius),
            )
        return None

    def _run(self) -> None:
        last_inference_started = 0.0
        while True:
            with self._condition:
                while not self._stop and self._request is None:
                    self._condition.wait()
                if self._stop:
                    return
                request = self._request
                self._request = None
            assert request is not None
            frame_id, timestamp_ns, rectified, urgent = request
            rate = self.reacquire_hz if urgent else self.normal_hz
            delay = inference_start_delay(
                last_inference_started, rate, time.monotonic()
            )
            if delay > 0:
                time.sleep(delay)
                with self._condition:
                    if self._request is not None:
                        frame_id, timestamp_ns, rectified, urgent = self._request
                        self._request = None
            last_inference_started = time.monotonic()
            try:
                candidate = self._infer(rectified)
                result = YoloResult(
                    frame_id, timestamp_ns, time.time_ns(), candidate, None
                )
            except Exception as error:
                result = YoloResult(
                    frame_id, timestamp_ns, time.time_ns(), None, str(error)
                )
            with self._condition:
                self._result = result


def inference_start_delay(last_started: float, rate_hz: float, now: float) -> float:
    """Rate-limit inference start times without adding inference time twice."""
    if rate_hz <= 0:
        raise ValueError("YOLO rate must be positive")
    if last_started <= 0:
        return 0.0
    return max(0.0, 1.0 / rate_hz - (now - last_started))
