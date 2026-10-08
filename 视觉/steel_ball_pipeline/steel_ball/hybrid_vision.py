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
        x, y, width, height = (float(roi[key]) for key in ("x", "y", "width", "height"))
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
            [mapped[:, 0].min(), mapped[:, 1].min(), mapped[:, 0].max(), mapped[:, 1].max()],
            dtype=np.float32,
        )

    def position_cm(
        self, candidate: BallCandidate, calibration: TrackCalibration
    ) -> tuple[float, float, float]:
        frame_x, frame_y = self.rectified_to_frame(candidate.center_x, candidate.center_y)
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


class CvBallDetector:
    def __init__(self, config: dict) -> None:
        self.expected_diameter = float(config.get("expected_diameter_px", 19.0))
        self.min_diameter = float(config.get("min_diameter_px", 8.0))
        self.max_diameter = float(config.get("max_diameter_px", 38.0))
        self.track_y_min = float(config.get("track_y_min", 20.0))
        self.track_y_max = float(config.get("track_y_max", 90.0))
        self.min_circularity = float(config.get("min_circularity", 0.18))
        self.min_confidence = float(config.get("min_confidence", 0.42))
        self.max_saturation = int(config.get("max_saturation", 110))
        self.prediction_gate_px = float(config.get("prediction_gate_px", 55.0))

    def detect(
        self, image: np.ndarray, predicted_center: tuple[float, float] | None = None
    ) -> BallCandidate | None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("CV detector expects a BGR image")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        local_background = cv2.GaussianBlur(smooth, (0, 0), 9.0)
        dark_difference = cv2.subtract(local_background, smooth)
        percentile = float(np.percentile(dark_difference, 92))
        contrast_threshold = int(np.clip(percentile * 0.55, 7, 35))
        local_mask = dark_difference >= contrast_threshold
        otsu_value, otsu_mask = cv2.threshold(
            smooth, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        if otsu_value < 40:
            otsu_mask[:] = 0
        saturation_mask = hsv[..., 1] <= self.max_saturation
        mask = np.where(
            saturation_mask & (local_mask | (otsu_mask > 0)), 255, 0
        ).astype(np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: BallCandidate | None = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 12.0:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            diameter = float(max(width, height))
            if not self.min_diameter <= diameter <= self.max_diameter:
                continue
            if not self.track_y_min <= y + height / 2 <= self.track_y_max:
                continue
            aspect = min(width, height) / max(width, height)
            if aspect < 0.42:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 4.0 * np.pi * area / max(1.0, perimeter**2)
            if circularity < self.min_circularity:
                continue
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            size_score = float(
                np.exp(-0.5 * ((2.0 * radius - self.expected_diameter) / 8.0) ** 2)
            )
            circularity_score = float(np.clip(circularity / 0.75, 0.0, 1.0))
            aspect_score = float(np.clip((aspect - 0.42) / 0.58, 0.0, 1.0))
            contrast_score = float(
                np.clip(np.mean(dark_difference[mask == 255]) / 45.0, 0.0, 1.0)
            ) if np.any(mask == 255) else 0.0
            center_line = (self.track_y_min + self.track_y_max) / 2.0
            y_score = float(
                np.clip(1.0 - abs(center_y - center_line) / max(1.0, self.track_y_max - self.track_y_min), 0.0, 1.0)
            )
            prediction_score = 0.5
            if predicted_center is not None:
                distance = float(
                    np.hypot(center_x - predicted_center[0], center_y - predicted_center[1])
                )
                if distance > self.prediction_gate_px:
                    continue
                prediction_score = float(np.clip(1.0 - distance / self.prediction_gate_px, 0.0, 1.0))
            confidence = (
                0.26 * size_score
                + 0.22 * circularity_score
                + 0.12 * aspect_score
                + 0.14 * contrast_score
                + 0.10 * y_score
                + 0.16 * prediction_score
            )
            candidate = BallCandidate(
                float(center_x),
                float(center_y),
                np.asarray([x, y, x + width, y + height], dtype=np.float32),
                float(confidence),
                "cv",
                float(radius),
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate
        if best is None or best.confidence < self.min_confidence:
            return None
        return best


class AsyncYoloWorker:
    def __init__(self, runtime: object, config: dict) -> None:
        self.runtime = runtime
        self.input_size = int(config.get("input_size", 640))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.03))
        self.iou_threshold = float(config.get("iou_threshold", 0.45))
        self.track_y_min = float(config.get("track_y_min", 0))
        self.track_y_max = float(config.get("track_y_max", 110))
        self.normal_hz = float(config.get("normal_hz", 30.0))
        self.reacquire_hz = float(config.get("reacquire_hz", 50.0))
        self._condition = threading.Condition()
        self._request: tuple[int, int, np.ndarray, bool] | None = None
        self._result: YoloResult | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="rknn-yolo", daemon=True)
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
    """Rate-limit inference start times without adding inference latency to the period."""
    if rate_hz <= 0:
        raise ValueError("YOLO rate must be positive")
    if last_started <= 0:
        return 0.0
    return max(0.0, 1.0 / rate_hz - (now - last_started))


class HybridSelector:
    def __init__(self, config: dict) -> None:
        self.yolo_max_age_ms = float(config.get("yolo_max_age_ms", 60.0))
        self.agreement_px = float(config.get("agreement_px", 28.0))
        self.cv_standalone_confidence = float(
            config.get("cv_standalone_confidence", 0.66)
        )

    def choose(
        self,
        cv_candidate: BallCandidate | None,
        yolo_result: YoloResult | None,
        timestamp_ns: int,
        tracker_available: bool,
        yolo_is_new: bool,
    ) -> tuple[BallCandidate | None, float]:
        yolo_candidate = None
        yolo_age_ms = float("inf")
        if yolo_result is not None and yolo_result.error is None:
            yolo_age_ms = max(0.0, (timestamp_ns - yolo_result.timestamp_ns) / 1e6)
            if yolo_age_ms <= self.yolo_max_age_ms:
                yolo_candidate = yolo_result.candidate
        if cv_candidate is not None and yolo_candidate is not None:
            distance = float(
                np.hypot(
                    cv_candidate.center_x - yolo_candidate.center_x,
                    cv_candidate.center_y - yolo_candidate.center_y,
                )
            )
            if distance <= self.agreement_px:
                return (
                    BallCandidate(
                        cv_candidate.center_x,
                        cv_candidate.center_y,
                        cv_candidate.box,
                        max(cv_candidate.confidence, yolo_candidate.confidence),
                        "cv+yolo",
                        cv_candidate.radius,
                    ),
                    0.08,
                )
            if yolo_is_new:
                return yolo_candidate, 0.18 + min(0.15, yolo_age_ms / 400.0)
            return None, 0.2
        if cv_candidate is not None:
            if tracker_available or cv_candidate.confidence >= self.cv_standalone_confidence:
                return cv_candidate, 0.12
            return None, 0.2
        if yolo_candidate is not None and yolo_is_new:
            return yolo_candidate, 0.18 + min(0.15, yolo_age_ms / 400.0)
        return None, 0.2
