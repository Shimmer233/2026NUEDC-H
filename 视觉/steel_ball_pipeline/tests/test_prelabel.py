from __future__ import annotations

import cv2
import numpy as np

from steel_ball.prelabel import TrackBallPrelabeler, yolo_line


def make_track(ball_center: tuple[int, int] | None = None) -> np.ndarray:
    image = np.full((110, 470), 35, dtype=np.uint8)
    cv2.rectangle(image, (5, 25), (464, 84), 220, -1)
    for x in range(20, 460, 20):
        cv2.line(image, (x, 27), (x, 32), 115, 2)
    if ball_center is not None:
        cv2.circle(image, ball_center, 11, 45, -1)
        cv2.circle(image, (ball_center[0] - 3, ball_center[1] - 3), 3, 190, -1)
    return image


def test_locates_dark_circle_on_track() -> None:
    background = make_track()
    current = make_track((217, 54))
    detector = TrackBallPrelabeler(background)
    detection = detector.locate(current)
    assert abs(detection.center_x - 217) <= 3
    assert abs(detection.center_y - 54) <= 3
    assert 10 <= detection.radius <= 13


def test_yolo_label_has_five_valid_fields() -> None:
    detector = TrackBallPrelabeler(make_track())
    detection = detector.locate(make_track((217, 54)))
    fields = yolo_line(detection, 470, 110).split()
    assert len(fields) == 5
    assert fields[0] == "0"
    assert all(0 < float(value) <= 1 for value in fields[1:])

