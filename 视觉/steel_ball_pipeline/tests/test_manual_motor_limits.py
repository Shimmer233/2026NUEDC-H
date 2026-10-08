from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.manual_motor_limits as manual


class FakeBus:
    def __init__(self, position: int) -> None:
        self.position = position
        self.enabled = False
        self.targets: list[int] = []
        self.estopped = False
        self.closed = False

    def read_position(self) -> int:
        return self.position

    def set_position_mode(self) -> None:
        return None

    def move_absolute(self, target: int, _speed: int, _acceleration: int) -> None:
        self.targets.append(target)
        self.position = target

    def enable(self, enabled: bool) -> None:
        self.enabled = enabled

    def read_system_status(self):
        return SimpleNamespace(
            actual_position=self.position,
            enabled=self.enabled,
            arrived=True,
            stalled=False,
        )

    def emergency_stop(self) -> None:
        self.estopped = True

    def close(self) -> None:
        self.closed = True


def base_args(tmp_path, **changes):
    values = {
        "device": "/dev/ttyS0",
        "baudrate": 115200,
        "address": 1,
        "timeout": 0.2,
        "state": tmp_path / "manual_motor_limits.json",
        "speed_rpm": 10,
        "acceleration": 10,
        "tolerance_counts": 5,
        "move_timeout": 1.0,
        "max_excursion_counts": 200,
        "confirm_level": True,
        "confirm_clearance": True,
        "arm": True,
        "offset_counts": 0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_capture_center_records_current_count_and_holds(tmp_path, monkeypatch) -> None:
    bus = FakeBus(43204)
    monkeypatch.setattr(manual, "connection", lambda _args: bus)

    assert manual.capture_center(base_args(tmp_path)) == 0

    saved = manual.ManualMotorLimits.load(tmp_path / "manual_motor_limits.json")
    assert saved.center_count == 43204
    assert saved.max_excursion_counts == 200
    assert bus.targets == [43204]
    assert bus.enabled


def test_move_uses_offset_from_saved_center(tmp_path, monkeypatch) -> None:
    record = manual.ManualMotorLimits(1, 1000, 200, "now")
    record.save(tmp_path / "manual_motor_limits.json")
    bus = FakeBus(1000)
    monkeypatch.setattr(manual, "connection", lambda _args: bus)

    args = base_args(tmp_path, offset_counts=-75)
    assert manual.move(args) == 0
    assert bus.targets == [925]
    assert bus.enabled


def test_move_rejects_stale_center_before_sending_motion(tmp_path, monkeypatch) -> None:
    record = manual.ManualMotorLimits(1, 1000, 200, "now")
    record.save(tmp_path / "manual_motor_limits.json")
    bus = FakeBus(5000)
    monkeypatch.setattr(manual, "connection", lambda _args: bus)

    with pytest.raises(RuntimeError, match="outside the saved center guard"):
        manual.move(base_args(tmp_path, offset_counts=50))
    assert bus.targets == []


def test_records_up_and_down_counts_with_phone_angles(tmp_path, monkeypatch) -> None:
    path = tmp_path / "manual_motor_limits.json"
    manual.ManualMotorLimits(1, 1000, 200, "now").save(path)
    bus = FakeBus(1080)
    monkeypatch.setattr(manual, "connection", lambda _args: bus)

    up_args = base_args(tmp_path, side="up", angle_deg=1.25)
    assert manual.record_limit(up_args) == 0
    bus.position = 910
    down_args = base_args(tmp_path, side="down", angle_deg=-1.4)
    assert manual.record_limit(down_args) == 0

    saved = manual.ManualMotorLimits.load(path)
    assert saved.up["count"] == 1080
    assert saved.up["offset_counts"] == 80
    assert saved.up["angle_deg"] == 1.25
    assert saved.down["count"] == 910
    assert saved.down["offset_counts"] == -90
    assert saved.down["angle_deg"] == -1.4
