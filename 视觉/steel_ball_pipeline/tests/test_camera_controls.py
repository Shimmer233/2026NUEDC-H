from __future__ import annotations

from steel_ball.camera_controls import (
    V4L2Control,
    parse_v4l2_controls,
    safe_exposure_absolute_max,
)


def test_parses_integer_and_menu_v4l2_controls() -> None:
    output = """
                     gain 0x00980913 (int)    : min=0 max=255 step=1 default=0 value=64
            exposure_auto 0x009a0901 (menu)   : min=0 max=3 step=1 default=3 value=1 (Manual Mode)
        exposure_absolute 0x009a0902 (int)    : min=3 max=2047 step=1 default=166 value=75
    """
    controls = parse_v4l2_controls(output)
    assert controls["gain"].maximum == 255
    assert controls["exposure_auto"].value == 1
    assert controls["exposure_absolute"].default == 166


def test_120_fps_exposure_is_limited_below_frame_period() -> None:
    control = V4L2Control("exposure_absolute", 3, 2047, 1, 166, 166)
    assert safe_exposure_absolute_max(control, 120.0) == 75
