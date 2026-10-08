from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


@dataclass(frozen=True)
class TrackCalibration:
    degree: int
    coefficients: np.ndarray
    origin_px: np.ndarray
    axis: np.ndarray
    points: tuple[tuple[float, float, float], ...]
    rmse_cm: float
    kind: str = "polynomial"
    length_cm: float | None = None
    pipe_corners_px: tuple[tuple[float, float], ...] = ()
    cm_per_px: float | None = None

    def position_cm(self, x_px: float, y_px: float) -> float:
        projected = (np.asarray([x_px, y_px]) - self.origin_px) @ self.axis
        if self.kind == "pipe_centerline" and self.cm_per_px is not None:
            return float(projected * self.cm_per_px)
        return float(np.polyval(self.coefficients, projected))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "kind": self.kind,
            "degree": self.degree,
            "coefficients": self.coefficients.tolist(),
            "origin_px": self.origin_px.tolist(),
            "axis": self.axis.tolist(),
            "rmse_cm": self.rmse_cm,
            "points": [
                {"x_px": float(x), "y_px": float(y), "s_cm": float(s)}
                for x, y, s in self.points
            ],
        }
        if self.kind == "pipe_centerline":
            payload.update(
                {
                    "length_cm": self.length_cm,
                    "cm_per_px": self.cm_per_px,
                    "pipe_corners_px": [
                        {"x_px": float(x), "y_px": float(y)}
                        for x, y in self.pipe_corners_px
                    ],
                }
            )
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TrackCalibration":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        kind = str(payload.get("kind", "polynomial"))
        points = tuple(
            (float(item["x_px"]), float(item["y_px"]), float(item["s_cm"]))
            for item in payload.get("points", [])
        )
        pipe_corners = tuple(
            (
                float(item["x_px"] if isinstance(item, dict) else item[0]),
                float(item["y_px"] if isinstance(item, dict) else item[1]),
            )
            for item in payload.get("pipe_corners_px", [])
        )
        return cls(
            degree=int(payload["degree"]),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            origin_px=np.asarray(payload["origin_px"], dtype=np.float64),
            axis=np.asarray(payload["axis"], dtype=np.float64),
            points=points,
            rmse_cm=float(payload["rmse_cm"]),
            kind=kind,
            length_cm=(
                None if payload.get("length_cm") is None else float(payload["length_cm"])
            ),
            pipe_corners_px=pipe_corners,
            cm_per_px=(
                None if payload.get("cm_per_px") is None else float(payload["cm_per_px"])
            ),
        )


def fit_track_calibration(
    points: Sequence[tuple[float, float, float]], degree: int = 2
) -> TrackCalibration:
    if degree not in (1, 2):
        raise ValueError("calibration degree must be 1 or 2")
    minimum = degree + 1
    if len(points) < minimum:
        raise ValueError(f"degree {degree} calibration requires at least {minimum} points")
    values = np.asarray(points, dtype=np.float64)
    pixel_points = values[:, :2]
    origin = pixel_points.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(pixel_points - origin, full_matrices=False)
    if singular_values[0] < 1.0:
        raise ValueError("calibration marks must span the track")
    axis = vh[0]
    projected = (pixel_points - origin) @ axis
    coefficients = np.polyfit(projected, values[:, 2], degree)
    if np.corrcoef(projected, values[:, 2])[0, 1] < 0:
        axis = -axis
        projected = -projected
        coefficients = np.polyfit(projected, values[:, 2], degree)
    residuals = np.polyval(coefficients, projected) - values[:, 2]
    rmse = float(np.sqrt(np.mean(residuals**2)))
    serializable_points = tuple(tuple(float(value) for value in row) for row in values)
    return TrackCalibration(degree, coefficients, origin, axis, serializable_points, rmse)


def fit_pipe_centerline_calibration(
    corners_px: Sequence[tuple[float, float]], length_cm: float = 25.0
) -> TrackCalibration:
    """Fit a signed one-dimensional pipe ruler from four clicked pipe corners.

    Corner order is fixed: left-top, left-bottom, right-bottom, right-top.
    The returned position is measured along the pipe centerline with the
    physical pipe center as 0 cm.
    """
    if length_cm <= 0:
        raise ValueError("pipe length must be positive")
    corners = np.asarray(corners_px, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError("pipe calibration requires four corner points")

    left_mid = (corners[0] + corners[1]) * 0.5
    right_mid = (corners[2] + corners[3]) * 0.5
    origin = (left_mid + right_mid) * 0.5
    axis_vector = right_mid - left_mid
    pixel_length = float(np.linalg.norm(axis_vector))
    if pixel_length < 1.0:
        raise ValueError("pipe corner points must span the 25 cm ruler")

    axis = axis_vector / pixel_length
    cm_per_px = float(length_cm / pixel_length)
    half_length = float(length_cm * 0.5)
    coefficients = np.asarray([cm_per_px, 0.0], dtype=np.float64)
    points = (
        (float(left_mid[0]), float(left_mid[1]), -half_length),
        (float(origin[0]), float(origin[1]), 0.0),
        (float(right_mid[0]), float(right_mid[1]), half_length),
    )
    serializable_corners = tuple(
        (float(point[0]), float(point[1])) for point in corners
    )
    return TrackCalibration(
        degree=1,
        coefficients=coefficients,
        origin_px=origin,
        axis=axis,
        points=points,
        rmse_cm=0.0,
        kind="pipe_centerline",
        length_cm=float(length_cm),
        pipe_corners_px=serializable_corners,
        cm_per_px=cm_per_px,
    )
