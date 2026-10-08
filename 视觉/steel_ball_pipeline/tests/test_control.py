from __future__ import annotations

from pathlib import Path

from steel_ball.control import (
    ActuatorCalibration,
    BalanceController,
    ControlState,
    ControllerGains,
    fit_actuator_calibration,
    fit_state_feedback_gains,
)


def calibration(verified: bool = True) -> ActuatorCalibration:
    return fit_actuator_calibration(
        [(-0.5, -500), (0.0, 0), (0.5, 500)],
        -800,
        800,
        persistence_verified=verified,
    )


def test_actuator_calibration_round_trip_and_persistence_gate(tmp_path: Path) -> None:
    value = calibration(False)
    assert not value.ready
    assert value.mapping_ready
    assert value.angle_to_count(0.25) == 250
    assert abs(value.count_to_angle(-250) + 0.25) < 1e-9
    path = tmp_path / "actuator.yaml"
    value.save(path)
    loaded = ActuatorCalibration.load(path)
    assert loaded == value


def test_controller_slews_target_limits_output_and_returns_zero_when_lost() -> None:
    controller = BalanceController(
        calibration(),
        ControllerGains(0.2, 0.02, 0.01, True),
        initial_target_cm=12.5,
        target_slew_cm_s=5.0,
        max_angle_deg=0.5,
    )
    controller.set_target(20.0)
    controller.update(12.5, 0.0, 0.0, True, False, True)
    output = controller.update(10.0, -2.0, 0.1, True, False, True)
    assert abs(output.target_cm - 13.0) < 1e-9
    assert output.angle_deg == 0.5
    assert output.saturated
    assert output.state == ControlState.ARMED
    lost = controller.update(None, None, 0.2, False, False, True)
    assert lost.state == ControlState.LOST
    assert lost.angle_deg == 0.0
    assert lost.target_count == 0


def test_state_feedback_fit_preserves_plant_direction() -> None:
    positive = fit_state_feedback_gains(10.0, 0.2)
    negative = fit_state_feedback_gains(-10.0, 0.2)
    assert positive.identified and negative.identified
    assert positive.kp_deg_per_cm > 0
    assert negative.kp_deg_per_cm < 0
