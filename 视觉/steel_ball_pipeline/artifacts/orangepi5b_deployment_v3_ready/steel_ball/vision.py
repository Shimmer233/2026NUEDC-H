from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxMeta:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scale: float
    pad_x: int
    pad_y: int


def letterbox_black(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize while preserving aspect ratio and pad with the RKNN demo's black border."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    output = np.zeros((size, size, 3), dtype=image.dtype)
    output[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    meta = LetterboxMeta(width, height, size, size, scale, pad_x, pad_y)
    return output, meta


def unletterbox_box(box_xyxy: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    box = np.asarray(box_xyxy, dtype=np.float32).copy()
    box[[0, 2]] = (box[[0, 2]] - meta.pad_x) / meta.scale
    box[[1, 3]] = (box[[1, 3]] - meta.pad_y) / meta.scale
    box[[0, 2]] = np.clip(box[[0, 2]], 0, meta.source_width - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, meta.source_height - 1)
    return box


def refine_ball_center(image: np.ndarray, box_xyxy: np.ndarray) -> tuple[float, float, str]:
    """Refine a detector center with an edge circle/ellipse; fall back to box center."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float32)
    fallback = ((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2, "box")
    margin = max(3, int(round(0.15 * max(x2 - x1, y2 - y1))))
    left = max(0, int(np.floor(x1)) - margin)
    top = max(0, int(np.floor(y1)) - margin)
    right = min(width, int(np.ceil(x2)) + margin + 1)
    bottom = min(height, int(np.ceil(y2)) + margin + 1)
    if right - left < 8 or bottom - top < 8:
        return fallback

    crop = image[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    expected_radius = max(3.0, 0.25 * ((x2 - x1) + (y2 - y1)))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(4, int(expected_radius)),
        param1=60,
        param2=8,
        minRadius=max(2, int(expected_radius * 0.55)),
        maxRadius=max(3, int(expected_radius * 1.45)),
    )
    target_x = fallback[0] - left
    target_y = fallback[1] - top
    if circles is not None:
        candidates = circles[0]
        distances = (candidates[:, 0] - target_x) ** 2 + (candidates[:, 1] - target_y) ** 2
        circle = candidates[int(np.argmin(distances))]
        return float(circle[0] + left), float(circle[1] + top), "circle"

    edges = cv2.Canny(gray, 30, 90)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ellipses: list[tuple[float, tuple[float, float]]] = []
    for contour in contours:
        if len(contour) < 5:
            continue
        (cx, cy), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        if min(axis_a, axis_b) < expected_radius or max(axis_a, axis_b) > expected_radius * 3.0:
            continue
        aspect = min(axis_a, axis_b) / max(axis_a, axis_b)
        if aspect < 0.55:
            continue
        distance = (cx - target_x) ** 2 + (cy - target_y) ** 2
        ellipses.append((distance, (cx, cy)))
    if ellipses:
        _, (cx, cy) = min(ellipses, key=lambda item: item[0])
        return float(cx + left), float(cy + top), "ellipse"
    return fallback
