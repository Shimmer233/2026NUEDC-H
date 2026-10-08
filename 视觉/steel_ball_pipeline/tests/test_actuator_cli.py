from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.calibrate_actuator as actuator_cli
from steel_ball.control import ActuatorCalibration


class PositionOnlyConnection:
    def __init__(self, position: int) -> None:
        self.position = position

    def read_position(self) -> int:
        return self.position

    def close(self) -> None:
        return None


class SessionZeroConnection(PositionOnlyConnection):
    def enable(self, enabled: bool) -> None:
        assert not enabled

    def set_position_mode(self) -> None:
        return None

    def zero_position(self) -> None:
        self.position = 0


def unverified_calibration(path) -> None:
    ActuatorCalibration(
        address=1,
        zero_count=0,
        counts_per_degree=1000.0,
        min_count=-500,
        max_count=500,
        zero_tolerance_counts=20,
        persistence_verified=False,
        points=((-0.5, -500), (0.0, 0), (0.5, 500)),
        calibrated_at="test",
    ).save(path)


def test_power_cycle_verification_can_promote_unverified_zero(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "actuator.yaml"
    unverified_calibration(output)
    monkeypatch.setattr(
        actuator_cli, "connection", lambda _args: PositionOnlyConnection(5)
    )
    args = SimpleNamespace(
        confirm_power_cycled=True,
        confirm_not_moved=True,
        output=output,
        session_marker=tmp_path / "session-zero.json",
    )
    assert actuator_cli.verify(args) == 0
    assert ActuatorCalibration.load(output).persistence_verified


def test_jog_is_blocked_without_persistence_or_session_zero(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "actuator.yaml"
    unverified_calibration(output)
    monkeypatch.setattr(
        actuator_cli, "connection", lambda _args: PositionOnlyConnection(0)
    )
    args = SimpleNamespace(
        arm=True,
        output=output,
        target_count=100,
        session_marker=tmp_path / "missing-session-zero.json",
    )
    with pytest.raises(RuntimeError, match="session zero is missing"):
        actuator_cli.jog(args)


def test_session_zero_preserves_angle_mapping_and_creates_marker(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "actuator.yaml"
    unverified_calibration(output)
    marker = tmp_path / "session-zero.json"
    monkeypatch.setattr(
        actuator_cli, "connection", lambda _args: SessionZeroConnection(35921)
    )
    monkeypatch.setattr(
        actuator_cli,
        "create_session_zero_marker",
        lambda path, **_kwargs: SimpleNamespace(boot_id="boot-test"),
    )
    args = SimpleNamespace(
        confirm_level=True,
        confirm_supported=True,
        output=output,
        min_count=None,
        max_count=None,
        zero_tolerance=None,
        session_marker=marker,
        address=1,
    )
    assert actuator_cli.session_zero(args) == 0
    calibration = ActuatorCalibration.load(output)
    assert calibration.zero_count == 0
    assert calibration.counts_per_degree == 1000.0
    assert not calibration.persistence_verified
