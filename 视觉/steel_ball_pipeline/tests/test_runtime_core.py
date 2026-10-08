from __future__ import annotations

from pathlib import Path

import numpy as np

from steel_ball.calibration import TrackCalibration, fit_track_calibration
from steel_ball.rknn_postprocess import decode_yolov5_heads
from steel_ball.tracking import PositionVelocityKalman
from steel_ball.vision import letterbox_black, rknn_nhwc_batch, unletterbox_box


def test_letterbox_box_round_trip() -> None:
    image = np.zeros((110, 470, 3), dtype=np.uint8)
    _, meta = letterbox_black(image, 640)
    original = np.asarray([100.0, 30.0, 125.0, 55.0])
    encoded = original.copy()
    encoded[[0, 2]] = encoded[[0, 2]] * meta.scale + meta.pad_x
    encoded[[1, 3]] = encoded[[1, 3]] * meta.scale + meta.pad_y
    np.testing.assert_allclose(unletterbox_box(encoded, meta), original, atol=1e-4)


def test_rknn_input_has_static_nhwc_batch_dimension() -> None:
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    tensor = rknn_nhwc_batch(image)
    assert tensor.shape == (1, 640, 640, 3)
    assert tensor.dtype == np.uint8
    assert tensor.flags.c_contiguous


def test_decodes_single_class_rknn_heads() -> None:
    outputs = [
        np.zeros((1, 18, 80, 80), dtype=np.float32),
        np.zeros((1, 18, 40, 40), dtype=np.float32),
        np.zeros((1, 18, 20, 20), dtype=np.float32),
    ]
    outputs[0][0, 0:4, 20, 30] = 0.5
    outputs[0][0, 4, 20, 30] = 0.9
    outputs[0][0, 5, 20, 30] = 0.8
    detections = decode_yolov5_heads(outputs, confidence_threshold=0.25)
    assert len(detections) == 1
    assert abs(detections[0].confidence - 0.72) < 1e-6
    np.testing.assert_allclose(detections[0].box, [239.0, 157.5, 249.0, 170.5])


def test_calibration_handles_horizontal_marks_and_round_trips(tmp_path: Path) -> None:
    points = [(100 + position * 10, 240.0, float(position)) for position in range(11)]
    calibration = fit_track_calibration(points, degree=2)
    assert abs(calibration.position_cm(155.0, 240.0) - 5.5) < 1e-8
    path = tmp_path / "calibration.yaml"
    calibration.save(path)
    loaded = TrackCalibration.load(path)
    assert abs(loaded.position_cm(155.0, 240.0) - 5.5) < 1e-8


def test_kalman_predicts_only_three_missing_frames() -> None:
    tracker = PositionVelocityKalman(max_prediction_frames=3)
    assert tracker.step(1.0, 0.0).detected
    assert tracker.step(1.1, 0.01).detected
    for index in range(1, 4):
        estimate = tracker.step(None, 0.01 + index * 0.01)
        assert estimate.available
        assert estimate.predicted
        assert estimate.missing_frames == index
    lost = tracker.step(None, 0.05)
    assert not lost.available
    assert not lost.predicted
