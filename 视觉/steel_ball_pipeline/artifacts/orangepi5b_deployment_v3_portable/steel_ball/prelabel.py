from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    center_x: float
    center_y: float
    radius: float
    score: float
    peak_gap: float
    hough_distance: float | None
    exposure_offset: float
    uncertain: bool

    def to_dict(self) -> dict[str, float | bool | None]:
        return asdict(self)


class TrackBallPrelabeler:
    """Locate one dark steel ball on a fixed bright track without a neural model."""

    def __init__(
        self,
        background: np.ndarray,
        search_y_min: int = 25,
        search_y_max: int = 85,
        expected_radius: int = 10,
        min_radius: int = 6,
        max_radius: int = 15,
        box_padding: float = 2.5,
        uncertain_score: float = 2.5,
        uncertain_peak_gap: float = 0.5,
    ) -> None:
        if background.ndim != 2:
            raise ValueError("background must be a grayscale image")
        self.background = background.astype(np.uint8, copy=True)
        self.search_y_min = search_y_min
        self.search_y_max = search_y_max
        self.expected_radius = expected_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.box_padding = box_padding
        self.uncertain_score = uncertain_score
        self.uncertain_peak_gap = uncertain_peak_gap
        self._contrast_kernel = self._make_contrast_kernel(expected_radius)

    @staticmethod
    def _make_contrast_kernel(radius: int) -> np.ndarray:
        outer_radius = radius + 8
        yy, xx = np.ogrid[-outer_radius : outer_radius + 1, -outer_radius : outer_radius + 1]
        distance = np.sqrt(xx * xx + yy * yy)
        inner = distance <= radius
        annulus = (distance >= radius + 3) & (distance <= outer_radius)
        kernel = np.zeros(distance.shape, dtype=np.float32)
        kernel[inner] = -1.0 / float(inner.sum())
        kernel[annulus] = 1.0 / float(annulus.sum())
        return kernel

    @staticmethod
    def _robust_zscore(values: np.ndarray, sample: np.ndarray) -> np.ndarray:
        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)))
        scale = max(1.4826 * mad, 1e-3)
        return (values - median) / scale

    def _score_map(self, gray: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = gray.shape
        x_min, x_max = 10, width - 10
        band = np.s_[self.search_y_min : self.search_y_max, x_min:x_max]
        exposure_offset = float(
            np.median(self.background[band].astype(np.int16) - gray[band].astype(np.int16))
        )
        aligned = np.clip(gray.astype(np.float32) + exposure_offset, 0, 255)
        positive_delta = np.maximum(self.background.astype(np.float32) - aligned, 0)
        positive_delta = cv2.GaussianBlur(positive_delta, (0, 0), 2.0)
        contrast = cv2.filter2D(aligned, cv2.CV_32F, self._contrast_kernel)
        delta_score = cv2.boxFilter(positive_delta, cv2.CV_32F, (17, 17), normalize=True)
        score = 0.55 * self._robust_zscore(contrast, contrast[band])
        score += 0.45 * self._robust_zscore(delta_score, delta_score[band])
        masked = np.full(score.shape, -1e9, dtype=np.float32)
        masked[band] = score[band]
        return masked, exposure_offset

    def _nearest_hough_circle(
        self, gray: np.ndarray, center_x: float, center_y: float
    ) -> tuple[float, float, float, float] | None:
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=12,
            param1=70,
            param2=14,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        if circles is None:
            return None
        candidates = circles[0]
        distances = np.hypot(candidates[:, 0] - center_x, candidates[:, 1] - center_y)
        index = int(np.argmin(distances))
        if float(distances[index]) > self.expected_radius + 5:
            return None
        x, y, radius = (float(value) for value in candidates[index])
        return x, y, radius, float(distances[index])

    def locate(self, gray: np.ndarray) -> Detection:
        if gray.shape != self.background.shape:
            raise ValueError(f"expected shape {self.background.shape}, got {gray.shape}")
        score_map, exposure_offset = self._score_map(gray)
        _, best_score, _, best_location = cv2.minMaxLoc(score_map)
        center_x, center_y = (float(best_location[0]), float(best_location[1]))

        suppression = score_map.copy()
        radius = self.expected_radius + 8
        x0 = max(0, best_location[0] - radius)
        x1 = min(gray.shape[1], best_location[0] + radius + 1)
        y0 = max(0, best_location[1] - radius)
        y1 = min(gray.shape[0], best_location[1] + radius + 1)
        suppression[y0:y1, x0:x1] = -1e9
        _, second_score, _, _ = cv2.minMaxLoc(suppression)
        peak_gap = float(best_score - second_score)

        hough = self._nearest_hough_circle(gray, center_x, center_y)
        hough_distance: float | None = None
        if hough is not None:
            center_x, center_y, hough_radius, hough_distance = hough
            radius_value = float(np.clip(hough_radius + self.box_padding, 10.0, 13.0))
        else:
            radius_value = float(self.expected_radius + self.box_padding)

        uncertain = (
            best_score < self.uncertain_score
            or peak_gap < self.uncertain_peak_gap
            or hough is None
        )
        return Detection(
            center_x=center_x,
            center_y=center_y,
            radius=radius_value,
            score=float(best_score),
            peak_gap=peak_gap,
            hough_distance=hough_distance,
            exposure_offset=exposure_offset,
            uncertain=uncertain,
        )


def yolo_line(detection: Detection, width: int, height: int) -> str:
    x0 = max(0.0, detection.center_x - detection.radius)
    y0 = max(0.0, detection.center_y - detection.radius)
    x1 = min(float(width), detection.center_x + detection.radius)
    y1 = min(float(height), detection.center_y + detection.radius)
    center_x = ((x0 + x1) / 2.0) / width
    center_y = ((y0 + y1) / 2.0) / height
    box_width = (x1 - x0) / width
    box_height = (y1 - y0) / height
    return f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}\n"

