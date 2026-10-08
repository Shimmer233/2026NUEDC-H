from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.calibration import (  # noqa: E402
    TrackCalibration,
    fit_pipe_centerline_calibration,
)


class PipeCenterlineCalibrationTest(unittest.TestCase):
    def test_horizontal_pipe_maps_centered_coordinates(self) -> None:
        calibration = fit_pipe_centerline_calibration(
            [(0.0, 0.0), (0.0, 20.0), (250.0, 20.0), (250.0, 0.0)]
        )

        self.assertAlmostEqual(calibration.position_cm(0.0, 10.0), -12.5)
        self.assertAlmostEqual(calibration.position_cm(125.0, 10.0), 0.0)
        self.assertAlmostEqual(calibration.position_cm(250.0, 10.0), 12.5)
        self.assertAlmostEqual(calibration.position_cm(125.0, 999.0), 0.0)

    def test_tilted_pipe_uses_axis_projection_only(self) -> None:
        left_mid = np.asarray([10.0, 20.0])
        right_mid = np.asarray([210.0, 120.0])
        axis = right_mid - left_mid
        axis = axis / np.linalg.norm(axis)
        normal = np.asarray([-axis[1], axis[0]])
        half_width_px = 12.0
        corners = [
            tuple(left_mid - normal * half_width_px),
            tuple(left_mid + normal * half_width_px),
            tuple(right_mid + normal * half_width_px),
            tuple(right_mid - normal * half_width_px),
        ]
        calibration = fit_pipe_centerline_calibration(corners)
        target_cm = 5.0
        point = calibration.origin_px + calibration.axis * (
            target_cm / calibration.cm_per_px
        )
        offset_point = point + normal * 40.0

        self.assertAlmostEqual(calibration.position_cm(*point), target_cm)
        self.assertAlmostEqual(calibration.position_cm(*offset_point), target_cm)

    def test_new_yaml_round_trips(self) -> None:
        calibration = fit_pipe_centerline_calibration(
            [(0.0, 0.0), (0.0, 20.0), (250.0, 20.0), (250.0, 0.0)]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "track_calibration.yaml"
            calibration.save(path)
            loaded = TrackCalibration.load(path)

        self.assertEqual(loaded.kind, "pipe_centerline")
        self.assertEqual(len(loaded.pipe_corners_px), 4)
        self.assertAlmostEqual(loaded.position_cm(250.0, 10.0), 12.5)

    def test_legacy_yaml_without_kind_still_loads(self) -> None:
        payload = {
            "version": 1,
            "degree": 1,
            "coefficients": [0.1, 0.0],
            "origin_px": [0.0, 0.0],
            "axis": [1.0, 0.0],
            "rmse_cm": 0.0,
            "points": [
                {"x_px": 0.0, "y_px": 0.0, "s_cm": 0.0},
                {"x_px": 100.0, "y_px": 0.0, "s_cm": 10.0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            loaded = TrackCalibration.load(path)

        self.assertEqual(loaded.kind, "polynomial")
        self.assertAlmostEqual(loaded.position_cm(50.0, 999.0), 5.0)


if __name__ == "__main__":
    unittest.main()
