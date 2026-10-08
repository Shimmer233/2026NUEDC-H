from __future__ import annotations

import cv2
import numpy as np

from steel_ball.hybrid_vision import (
    BallCandidate,
    CvBallDetector,
    HybridSelector,
    TrackGeometry,
    TrackStabilizer,
    YoloResult,
    inference_start_delay,
)


def test_track_geometry_round_trip() -> None:
    geometry = TrackGeometry(
        np.asarray([[10, 20], [480, 25], [475, 135], [15, 130]], dtype=np.float32),
        (470, 110),
    )
    frame_point = geometry.rectified_to_frame(235.0, 55.0)
    rectified = geometry.frame_to_rectified(*frame_point)
    np.testing.assert_allclose(rectified, (235.0, 55.0), atol=1e-3)


def test_cv_detector_finds_dark_ball_and_rejects_colored_marks() -> None:
    image = np.full((110, 470, 3), 225, dtype=np.uint8)
    for x in range(30, 450, 40):
        cv2.rectangle(image, (x, 46), (x + 5, 64), (220, 40, 40), -1)
    cv2.circle(image, (210, 55), 10, (45, 55, 60), -1)
    detector = CvBallDetector(
        {
            "track_y_min": 25,
            "track_y_max": 85,
            "min_confidence": 0.35,
            "prediction_gate_px": 70,
        }
    )
    candidate = detector.detect(image, (208.0, 55.0))
    assert candidate is not None
    assert abs(candidate.center_x - 210.0) <= 2.0
    assert abs(candidate.center_y - 55.0) <= 2.0


def test_stabilizer_removes_small_camera_translation() -> None:
    reference = np.full((110, 470, 3), 210, dtype=np.uint8)
    cv2.rectangle(reference, (30, 25), (90, 80), (40, 40, 40), -1)
    cv2.line(reference, (140, 30), (410, 72), (80, 80, 80), 3)
    shifted = cv2.warpAffine(
        reference,
        np.asarray([[1, 0, 4], [0, 1, -3]], dtype=np.float32),
        (470, 110),
        borderMode=cv2.BORDER_REPLICATE,
    )
    stabilizer = TrackStabilizer(
        {"enabled": True, "max_shift_px": 8, "min_response": 0.01},
        (470, 110),
    )
    stabilizer.apply(reference)
    aligned = stabilizer.apply(shifted)
    assert abs(stabilizer.last_shift[0] - 4.0) < 0.5
    assert abs(stabilizer.last_shift[1] + 3.0) < 0.5
    assert np.mean(np.abs(aligned.astype(float) - reference.astype(float))) < 3.0


def test_selector_prefers_current_cv_when_yolo_agrees() -> None:
    box = np.asarray([90, 40, 110, 60], dtype=np.float32)
    cv_candidate = BallCandidate(100, 50, box, 0.7, "cv", 10)
    yolo_candidate = BallCandidate(103, 51, box, 0.8, "yolo", 10)
    yolo = YoloResult(4, 1_000_000_000, 1_010_000_000, yolo_candidate)
    selected, measurement_std = HybridSelector({}).choose(
        cv_candidate, yolo, 1_020_000_000, True, True
    )
    assert selected is not None
    assert selected.source == "cv+yolo"
    assert selected.center_x == 100
    assert measurement_std == 0.08


def test_yolo_rate_is_measured_between_inference_start_times() -> None:
    assert abs(inference_start_delay(10.0, 50.0, 10.013) - 0.007) < 1e-9
    assert inference_start_delay(10.0, 50.0, 10.025) == 0.0
