from __future__ import annotations

from board.steel_ball_tracker import (
    centered_coordinate_cm,
    coordinate_tick_values,
    make_controller,
)
from steel_ball.control import ActuatorCalibration, BalanceController


def actuator() -> ActuatorCalibration:
    return ActuatorCalibration(
        address=1,
        zero_count=0,
        counts_per_degree=1000.0,
        min_count=-1000,
        max_count=1000,
        zero_tolerance_counts=20,
        persistence_verified=True,
        points=((-0.5, -500), (0.0, 0), (0.5, 500)),
        calibrated_at="test",
    )


def test_centered_axis_matches_25_cm_pipe_definition() -> None:
    ticks = coordinate_tick_values()
    assert len(ticks) == 51
    assert ticks[0] == (0.0, -12.5)
    assert ticks[25] == (12.5, 0.0)
    assert ticks[-1] == (25.0, 12.5)
    assert centered_coordinate_cm(7.5) == -5.0


def test_make_controller_returns_configured_controller() -> None:
    config = {
        "control": {
            "target_min_cm": 1.0,
            "target_max_cm": 24.0,
            "target_slew_cm_s": 5.0,
            "max_angle_deg": 0.8,
            "integral_band_cm": 2.0,
            "integral_angle_limit_deg": 0.2,
            "gains": {
                "kp_deg_per_cm": 0.04,
                "kv_deg_per_cm_s": 0.02,
                "ki_deg_per_cm_s": 0.0032,
                "identified": False,
            },
        }
    }
    controller = make_controller(config, actuator(), 12.5)
    assert isinstance(controller, BalanceController)
    assert controller.requested_target_cm == 12.5
