from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .calibration import TrackCalibration
from .vision_runtime import BallCandidate, TrackGeometry


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
    track_length_cm: float = 25.0,
    center_cm: float = 12.5,
    step_cm: float = 0.5,
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

    detected_state = (
        "PRED" if predicted else candidate.source.upper() if candidate else "LOST"
    )
    center_cm = float(coordinate_config.get("center_cm", 12.5))
    position_text = (
        "--"
        if position_cm is None
        else f"{centered_coordinate_cm(position_cm, center_cm):+.3f} cm"
    )
    velocity_text = (
        "--" if velocity_cm_s is None else f"{velocity_cm_s:+.2f} cm/s"
    )
    lines = (
        f"{detected_state}  x={position_text}  v={velocity_text}",
        f"latency={latency_ms:.1f} ms  loop={loop_fps:.1f} fps  yolo={yolo_fps:.1f} fps",
    )
    for index, text in enumerate(lines):
        origin = (8, 22 + index * 22)
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.51,
            (0, 0, 0),
            3,
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
