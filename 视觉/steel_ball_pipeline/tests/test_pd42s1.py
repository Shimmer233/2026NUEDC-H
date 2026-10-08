from __future__ import annotations

import struct
import time

import pytest

from steel_ball.pd42s1 import (
    CMD_READ_POSITION,
    CMD_READ_SYSTEM,
    PD42S1CommandError,
    PD42S1Connection,
    PD42S1ProtocolError,
    build_frame,
    decode_position,
    decode_system_status,
    frame_absolute_position,
    frame_clear_state,
    frame_enable,
    frame_save_parameters,
    frame_set_position_mode,
    frame_stop_now,
    frame_zero_position,
    MotorWorker,
    SystemStatus,
    parse_response,
)


def test_manual_command_examples_are_exact() -> None:
    assert frame_save_parameters().hex(" ").upper() == "C5 01 04 CA 5C"
    assert frame_set_position_mode().hex(" ").upper() == "C5 01 62 00 28 5C"
    assert frame_zero_position().hex(" ").upper() == "C5 01 F8 BE 5C"
    assert frame_enable(True).hex(" ").upper() == "C5 01 FA 00 C0 5C"
    assert frame_enable(False).hex(" ").upper() == "C5 01 FA 01 C1 5C"
    assert frame_clear_state().hex(" ").upper() == "C5 01 FB C1 5C"
    assert frame_stop_now().hex(" ").upper() == "C5 01 FC C2 5C"
    assert frame_absolute_position(512000, 2000, 100).hex(" ").upper() == (
        "C5 01 F2 00 64 07 D0 00 07 D0 00 CA 5C"
    )


def test_absolute_position_encodes_negative_target_and_rejects_zero_acceleration() -> None:
    frame = frame_absolute_position(-500, speed_rpm=10, acceleration=10)
    assert len(frame) == 13
    assert frame[3] == 1
    assert int.from_bytes(frame[7:11], "big") == 500
    with pytest.raises(ValueError, match="zero starts immediately"):
        frame_absolute_position(0, acceleration=0)


def test_response_parser_validates_checksum_and_error_result() -> None:
    reply = bytes.fromhex("C5 01 F8 01 BF 5C")
    assert parse_response(reply, 0xF8, 1).result == 1
    with pytest.raises(PD42S1ProtocolError, match="checksum"):
        parse_response(bytes.fromhex("C5 01 F8 01 00 5C"), 0xF8, 1)
    with pytest.raises(PD42S1CommandError) as error:
        parse_response(build_frame(1, 0xF8, bytes((0xE6,))), 0xF8, 1)
    assert error.value.code == 0xE6
    with pytest.raises(PD42S1ProtocolError, match="address"):
        parse_response(reply, 0xF8, 2)
    with pytest.raises(PD42S1ProtocolError, match="function"):
        parse_response(reply, 0xFA, 1)
    with pytest.raises(PD42S1ProtocolError, match="too short"):
        parse_response(reply[:5], 0xF8, 1)


def test_decodes_signed_position_and_full_system_status() -> None:
    position_reply = parse_response(
        build_frame(1, CMD_READ_POSITION, bytes((1,)) + (-321).to_bytes(4, "big", signed=True))
    )
    assert decode_position(position_reply) == -321
    payload = bytes((1,)) + struct.pack(
        ">fhfffhiiiIBBBB",
        24.5,
        120,
        1.2,
        2.3,
        4.5,
        -12,
        100,
        90,
        10,
        1234,
        0,
        1,
        0,
        0,
    )
    status = decode_system_status(parse_response(build_frame(1, CMD_READ_SYSTEM, payload)))
    assert abs(status.bus_voltage_v - 24.5) < 1e-5
    assert status.actual_position == 90
    assert status.position_error == 10
    assert status.enabled
    assert status.arrived
    assert not status.stalled


class FakeSerial:
    def __init__(self, response: bytes) -> None:
        self.response = bytearray(response)
        self.writes: list[bytes] = []
        self.closed = False

    def reset_input_buffer(self) -> None:
        return None

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        return None

    def read(self, size: int = 1) -> bytes:
        chunk = bytes(self.response[:size])
        del self.response[:size]
        return chunk

    def close(self) -> None:
        self.closed = True


class FragmentedSerial(FakeSerial):
    def read(self, size: int = 1) -> bytes:
        return super().read(min(size, 1))


def test_connection_writes_each_command_as_one_contiguous_frame() -> None:
    response = build_frame(
        1, CMD_READ_POSITION, bytes((1,)) + (42).to_bytes(4, "big", signed=True)
    )
    serial = FakeSerial(response)
    connection = PD42S1Connection(serial_port=serial)
    assert connection.read_position() == 42
    assert len(serial.writes) == 1
    assert serial.writes[0] == build_frame(1, CMD_READ_POSITION)


def test_connection_reassembles_fragmented_reply_and_times_out_on_residual_frame() -> None:
    response = build_frame(
        1, CMD_READ_POSITION, bytes((1,)) + (-7).to_bytes(4, "big", signed=True)
    )
    fragmented = FragmentedSerial(response)
    assert PD42S1Connection(serial_port=fragmented).read_position() == -7
    residual = FakeSerial(response[:4])
    with pytest.raises(PD42S1ProtocolError, match="serial timeout"):
        PD42S1Connection(serial_port=residual, timeout_seconds=0.001).read_position()


def system_status_response(*, enabled: bool) -> bytes:
    payload = bytes((1,)) + struct.pack(
        ">fhfffhiiiIBBBB",
        24.0,
        0,
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        0,
        0,
        0 if enabled else 1,
        1,
        0,
        0,
    )
    return build_frame(1, CMD_READ_SYSTEM, payload)


def test_enable_ack_is_followed_by_system_status_verification() -> None:
    ack = build_frame(1, 0xFA, bytes((1, 1)))
    serial = FakeSerial(ack + system_status_response(enabled=False))
    connection = PD42S1Connection(serial_port=serial)
    connection.enable(False)
    assert serial.writes == [frame_enable(False), build_frame(1, CMD_READ_SYSTEM)]


def test_enable_rejects_ack_when_system_status_did_not_change() -> None:
    ack = build_frame(1, 0xFA, bytes((1, 1)))
    serial = FakeSerial(ack + system_status_response(enabled=True))
    connection = PD42S1Connection(serial_port=serial, timeout_seconds=0.001)
    with pytest.raises(PD42S1ProtocolError, match="still reports enabled"):
        connection.enable(
            False,
            verification_timeout_seconds=0.003,
            verification_interval_seconds=0.0,
        )


def test_physically_verified_stale_status_accepts_disable_ack_only() -> None:
    ack = build_frame(1, 0xFA, bytes((1, 1)))
    serial = FakeSerial(ack)
    connection = PD42S1Connection(
        serial_port=serial,
        stale_disabled_status_physically_verified=True,
    )
    connection.enable(False)
    assert serial.writes == [frame_enable(False)]


def test_stale_disable_compatibility_does_not_weaken_enable_verification() -> None:
    ack = build_frame(1, 0xFA, bytes((1, 0)))
    serial = FakeSerial(ack + system_status_response(enabled=True))
    connection = PD42S1Connection(
        serial_port=serial,
        stale_disabled_status_physically_verified=True,
    )
    connection.enable(True)
    assert serial.writes == [frame_enable(True), build_frame(1, CMD_READ_SYSTEM)]


def system_status(*, position: int = 0, stalled: bool = False) -> SystemStatus:
    return SystemStatus(
        bus_voltage_v=24.0,
        phase_current_ma=0,
        flux_mwb=0.0,
        phase_resistance_ohm=0.0,
        phase_inductance_mh=0.0,
        speed_rpm=0,
        target_position=position,
        actual_position=position,
        position_error=0,
        pulse_count=0,
        enabled=True,
        arrived=True,
        stalled=stalled,
        group_address_mode=False,
    )


class FakeMotorConnection:
    def __init__(self, position: int = 0, stalled: bool = False) -> None:
        self.position = position
        self.stalled = stalled
        self.estop_calls = 0

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def enable(self, _enabled: bool) -> None:
        return None

    def move_absolute(self, target: int, _speed: int, _acceleration: int) -> None:
        self.position = target

    def read_position(self) -> int:
        return self.position

    def read_system_status(self) -> SystemStatus:
        return system_status(position=self.position, stalled=self.stalled)

    def emergency_stop(self) -> None:
        self.estop_calls += 1


@pytest.mark.parametrize(
    ("position", "stalled", "fault_text"),
    ((0, True, "stall"), (200, False, "outside soft limits")),
)
def test_motor_worker_immediately_latches_hardware_faults(
    position: int, stalled: bool, fault_text: str
) -> None:
    connection = FakeMotorConnection(position, stalled)
    worker = MotorWorker(connection, -100, 100, command_hz=200, status_hz=200)
    worker.start()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and not worker.snapshot().estop:
        time.sleep(0.002)
    snapshot = worker.snapshot()
    worker.close()
    assert snapshot.estop
    assert fault_text in str(snapshot.fault)
    assert connection.estop_calls == 1
