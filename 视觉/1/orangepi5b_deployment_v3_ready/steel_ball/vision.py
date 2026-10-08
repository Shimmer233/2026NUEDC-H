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


def rknn_nhwc_batch(image: np.ndarray) -> np.ndarray:
    """Add the static batch dimension expected by RKNNLite NHWC inputs."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HWC three-channel image, got {image.shape}")
    return np.ascontiguousarray(image[None, ...])


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


def _draw_labeled_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.46,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def _format_cm_label(value: float) -> str:
    if abs(value) < 1e-6:
        return "0"
    return f"{value:+.1f}".rstrip("0").rstrip(".")


def draw_pipe_ruler_overlay(
    image: np.ndarray,
    calibration: object | None,
    position_cm: float | None = None,
) -> None:
    """Draw a 25 cm centerline ruler for pipe-centerline calibration files."""
    if calibration is None or getattr(calibration, "kind", "") != "pipe_centerline":
        return
    corners = getattr(calibration, "pipe_corners_px", ())
    cm_per_px = getattr(calibration, "cm_per_px", None)
    length_cm = getattr(calibration, "length_cm", None)
    if len(corners) != 4 or cm_per_px is None or length_cm is None:
        return
    cm_per_px = float(cm_per_px)
    length_cm = float(length_cm)
    if cm_per_px <= 0 or length_cm <= 0:
        return

    corner_array = np.asarray(corners, dtype=np.float32)
    cv2.polylines(
        image,
        [corner_array.astype(np.int32)],
        isClosed=True,
        color=(255, 180, 0),
        thickness=1,
    )

    origin = np.asarray(getattr(calibration, "origin_px"), dtype=np.float64)
    axis = np.asarray(getattr(calibration, "axis"), dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return
    axis = axis / axis_norm
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    half_length_cm = length_cm * 0.5
    left = origin - axis * (half_length_cm / cm_per_px)
    right = origin + axis * (half_length_cm / cm_per_px)
    cv2.line(image, tuple(np.round(left).astype(int)), tuple(np.round(right).astype(int)), (0, 255, 255), 1)
    cv2.circle(image, tuple(np.round(origin).astype(int)), 4, (0, 255, 255), -1)

    whole_start = int(np.ceil(-half_length_cm))
    whole_end = int(np.floor(half_length_cm))
    tick_values = [-half_length_cm]
    tick_values.extend(float(value) for value in range(whole_start, whole_end + 1))
    tick_values.append(half_length_cm)
    seen: set[float] = set()
    for value_cm in tick_values:
        rounded_value = round(float(value_cm), 6)
        if rounded_value in seen:
            continue
        seen.add(rounded_value)
        point = origin + axis * (value_cm / cm_per_px)
        is_endpoint = abs(abs(value_cm) - half_length_cm) < 1e-6
        is_major = is_endpoint or abs(value_cm) < 1e-6 or abs(value_cm % 5.0) < 1e-6
        tick_half = 11.0 if is_major else 6.0
        start = point - normal * tick_half
        end = point + normal * tick_half
        cv2.line(
            image,
            tuple(np.round(start).astype(int)),
            tuple(np.round(end).astype(int)),
            (0, 255, 255),
            2 if is_major else 1,
        )

    for value_cm in (-half_length_cm, 0.0, half_length_cm):
        point = origin + axis * (value_cm / cm_per_px)
        label_point = point + normal * 24.0 + np.asarray([4.0, -4.0])
        _draw_labeled_text(
            image,
            _format_cm_label(value_cm),
            tuple(np.round(label_point).astype(int)),
            (0, 255, 255),
        )

    if position_cm is not None:
        marker = origin + axis * (float(position_cm) / cm_per_px)
        cv2.circle(image, tuple(np.round(marker).astype(int)), 6, (0, 165, 255), 2)
        label_point = marker - normal * 24.0 + np.asarray([6.0, 6.0])
        _draw_labeled_text(
            image,
            f"x={position_cm:+.2f} cm",
            tuple(np.round(label_point).astype(int)),
            (0, 165, 255),
        )
