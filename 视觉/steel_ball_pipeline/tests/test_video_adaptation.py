import numpy as np

from scripts.prepare_video_adaptation import (
    box_to_yolo,
    motion_expanded_box,
    temporal_appearance_box,
)
from steel_ball.prelabel import Detection


def make_detection(x: float, y: float) -> Detection:
    return Detection(
        center_x=x,
        center_y=y,
        radius=12.0,
        score=5.0,
        peak_gap=1.0,
        hough_distance=None,
        exposure_offset=0.0,
        uncertain=False,
    )


def test_motion_box_expands_horizontal_streak() -> None:
    background = np.full((110, 470), 220, dtype=np.uint8)
    frame = background.copy()
    frame[50:63, 180:224] = 70
    box, method, unusual = motion_expanded_box(frame, background, make_detection(202, 56), 25, 85)
    assert method == "motion_component"
    assert box[0] <= 180 and box[2] >= 224
    assert box[2] - box[0] > 40
    assert not unusual


def test_box_to_yolo_has_exactly_five_fields() -> None:
    fields = box_to_yolo((10, 20, 30, 40), 100, 50).split()
    assert fields[0] == "0"
    assert len(fields) == 5


def test_temporal_appearance_prefers_moving_streak_over_static_endpoint() -> None:
    current = np.full((110, 470), 220, dtype=np.uint8)
    previous = current.copy()
    following = current.copy()
    current[45:65, 195:225] = 90
    previous[45:65, 155:185] = 90
    following[45:65, 235:265] = 90
    for image in (previous, current, following):
        image[35:70, 414:428] = 70
    box, score, _, uncertain = temporal_appearance_box(current, previous, following)
    assert box[0] <= 195 and box[2] >= 225
    assert score > 80
    assert not uncertain
