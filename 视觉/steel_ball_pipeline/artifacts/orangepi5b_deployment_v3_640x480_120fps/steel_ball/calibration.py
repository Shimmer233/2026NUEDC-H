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

    def position_cm(self, x_px: float, y_px: float) -> float:
        projected = (np.asarray([x_px, y_px]) - self.origin_px) @ self.axis
        return float(np.polyval(self.coefficients, projected))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
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
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TrackCalibration":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        points = tuple(
            (float(item["x_px"]), float(item["y_px"]), float(item["s_cm"]))
            for item in payload["points"]
        )
        return cls(
            degree=int(payload["degree"]),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            origin_px=np.asarray(payload["origin_px"], dtype=np.float64),
            axis=np.asarray(payload["axis"], dtype=np.float64),
            points=points,
            rmse_cm=float(payload["rmse_cm"]),
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
