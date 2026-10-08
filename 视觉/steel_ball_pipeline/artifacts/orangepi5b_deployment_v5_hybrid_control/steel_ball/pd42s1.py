from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, replace
from typing import Protocol


FRAME_HEAD = 0xC5
FRAME_TAIL = 0x5C
ACK_SUCCESS = 0x01

CMD_SAVE_PARAMETERS = 0x04
CMD_READ_POSITION = 0x2A
CMD_READ_SYSTEM = 0x31
CMD_SET_MODE = 0x62
CMD_ABSOLUTE_POSITION = 0xF2
CMD_ZERO_POSITION = 0xF8
CMD_ENABLE = 0xFA
CMD_CLEAR_STATE = 0xFB
CMD_STOP_NOW = 0xFC

MODE_COMMUNICATION_POSITION = 0x00


class PD42S1Error(RuntimeError):
    pass


class PD42S1ProtocolError(PD42S1Error):
    pass


class PD42S1CommandError(PD42S1Error):
    ERROR_NAMES = {
        0xE1: "frame too short",
        0xE2: "invalid frame head",
        0xE3: "invalid frame tail",
        0xE4: "checksum mismatch",
        0xE5: "unsupported function",
        0xE6: "illegal data",
    }

    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(self.ERROR_NAMES.get(code, f"driver error 0x{code:02X}"))


class SerialPort(Protocol):
    def write(self, data: bytes) -> int: ...

    def read(self, size: int = 1) -> bytes: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ResponseFrame:
    address: int
    function: int
    data: bytes

    @property
    def result(self) -> int:
        if not self.data:
            raise PD42S1ProtocolError("response has no result byte")
        return self.data[0]


@dataclass(frozen=True)
class SystemStatus:
    bus_voltage_v: float
    phase_current_ma: int
    flux_mwb: float
    phase_resistance_ohm: float
    phase_inductance_mh: float
    speed_rpm: int
    target_position: int
    actual_position: int
    position_error: int
    pulse_count: int
    enabled: bool
    arrived: bool
    stalled: bool
    group_address_mode: bool


@dataclass(frozen=True)
class MotorSnapshot:
    connected: bool = False
    armed: bool = False
    actual_count: int | None = None
    target_count: int | None = None
    status: SystemStatus | None = None
    consecutive_errors: int = 0
    fault: str | None = None
    estop: bool = False
    timestamp_ns: int = 0


def checksum(data: bytes) -> int:
    return sum(data) & 0xFF


def build_frame(address: int, function: int, data: bytes = b"") -> bytes:
    if not 1 <= address <= 0xFF:
        raise ValueError("PD42S1 address must be in 1..255")
    if not 0 <= function <= 0xFF:
        raise ValueError("function must fit in one byte")
    body = bytes((FRAME_HEAD, address, function)) + bytes(data)
    return body + bytes((checksum(body), FRAME_TAIL))


def parse_response(
    raw: bytes, expected_function: int | None = None, expected_address: int | None = None
) -> ResponseFrame:
    if len(raw) < 6:
        raise PD42S1ProtocolError(f"response is too short: {len(raw)} bytes")
    if raw[0] != FRAME_HEAD or raw[-1] != FRAME_TAIL:
        raise PD42S1ProtocolError("invalid response frame boundary")
    if checksum(raw[:-2]) != raw[-2]:
        raise PD42S1ProtocolError("response checksum mismatch")
    address = raw[1]
    function = raw[2]
    if expected_address is not None and address != expected_address:
        raise PD42S1ProtocolError(
            f"response address 0x{address:02X} != 0x{expected_address:02X}"
        )
    if expected_function is not None and function != expected_function:
        raise PD42S1ProtocolError(
            f"response function 0x{function:02X} != 0x{expected_function:02X}"
        )
    frame = ResponseFrame(address, function, raw[3:-2])
    if frame.result != ACK_SUCCESS:
        raise PD42S1CommandError(frame.result)
    return frame


def frame_save_parameters(address: int = 1) -> bytes:
    return build_frame(address, CMD_SAVE_PARAMETERS)


def frame_set_position_mode(address: int = 1) -> bytes:
    return build_frame(address, CMD_SET_MODE, bytes((MODE_COMMUNICATION_POSITION,)))


def frame_zero_position(address: int = 1) -> bytes:
    return build_frame(address, CMD_ZERO_POSITION)


def frame_enable(enabled: bool, address: int = 1) -> bytes:
    return build_frame(address, CMD_ENABLE, bytes((0 if enabled else 1,)))


def frame_clear_state(address: int = 1) -> bytes:
    return build_frame(address, CMD_CLEAR_STATE)


def frame_stop_now(address: int = 1) -> bytes:
    return build_frame(address, CMD_STOP_NOW)


def frame_read_position(address: int = 1) -> bytes:
    return build_frame(address, CMD_READ_POSITION)


def frame_read_system(address: int = 1) -> bytes:
    return build_frame(address, CMD_READ_SYSTEM)


def frame_absolute_position(
    target_count: int,
    speed_rpm: int = 10,
    acceleration: int = 10,
    address: int = 1,
) -> bytes:
    if not -(2**31) <= target_count <= 2**31 - 1:
        raise ValueError("absolute position must fit in signed 32 bits")
    if not 1 <= speed_rpm <= 6000:
        raise ValueError("speed_rpm must be in 1..6000")
    if not 1 <= acceleration <= 200:
        raise ValueError("acceleration must be in 1..200; zero starts immediately")
    direction = 0 if target_count >= 0 else 1
    data = bytes((direction, acceleration))
    data += int(speed_rpm).to_bytes(2, "big", signed=False)
    data += abs(int(target_count)).to_bytes(4, "big", signed=False)
    return build_frame(address, CMD_ABSOLUTE_POSITION, data)


def decode_position(frame: ResponseFrame) -> int:
    if frame.function != CMD_READ_POSITION or len(frame.data) != 5:
        raise PD42S1ProtocolError("0x2A response must contain result + int32 position")
    return int.from_bytes(frame.data[1:5], "big", signed=True)


def decode_system_status(frame: ResponseFrame) -> SystemStatus:
    if frame.function != CMD_READ_SYSTEM or len(frame.data) != 41:
        raise PD42S1ProtocolError("0x31 response must contain 41 data bytes")
    values = struct.unpack(">fhfffhiiiIBBBB", frame.data[1:])
    return SystemStatus(
        bus_voltage_v=float(values[0]),
        phase_current_ma=int(values[1]),
        flux_mwb=float(values[2]),
        phase_resistance_ohm=float(values[3]),
        phase_inductance_mh=float(values[4]),
        speed_rpm=int(values[5]),
        target_position=int(values[6]),
        actual_position=int(values[7]),
        position_error=int(values[8]),
        pulse_count=int(values[9]),
        enabled=values[10] == 0,
        arrived=values[11] == 1,
        stalled=values[12] == 1,
        group_address_mode=values[13] == 1,
    )


class PD42S1Connection:
    RESPONSE_LENGTHS = {
        CMD_SAVE_PARAMETERS: 6,
        CMD_READ_POSITION: 10,
        CMD_READ_SYSTEM: 46,
        CMD_SET_MODE: 7,
        CMD_ABSOLUTE_POSITION: 6,
        CMD_ZERO_POSITION: 6,
        CMD_ENABLE: 7,
        CMD_CLEAR_STATE: 6,
        CMD_STOP_NOW: 6,
    }

    def __init__(
        self,
        device: str = "/dev/ttyS0",
        baudrate: int = 115200,
        address: int = 1,
        timeout_seconds: float = 0.2,
        stale_disabled_status_physically_verified: bool = False,
        serial_port: SerialPort | None = None,
    ) -> None:
        self.device = device
        self.baudrate = baudrate
        self.address = address
        self.timeout_seconds = timeout_seconds
        self.stale_disabled_status_physically_verified = bool(
            stale_disabled_status_physically_verified
        )
        self._serial = serial_port
        self._lock = threading.Lock()

    def open(self) -> None:
        if self._serial is not None:
            return
        try:
            import serial
        except ImportError as error:
            raise PD42S1Error("install pyserial before opening the motor UART") from error
        self._serial = serial.Serial(
            self.device,
            self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout_seconds,
            write_timeout=self.timeout_seconds,
        )

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _read_exactly(self, length: int) -> bytes:
        assert self._serial is not None
        deadline = time.monotonic() + self.timeout_seconds
        chunks = bytearray()
        while len(chunks) < length and time.monotonic() < deadline:
            chunk = self._serial.read(length - len(chunks))
            if chunk:
                chunks.extend(chunk)
        if len(chunks) != length:
            raise PD42S1ProtocolError(
                f"serial timeout: expected {length} response bytes, got {len(chunks)}"
            )
        return bytes(chunks)

    def transact(self, request: bytes) -> ResponseFrame:
        if len(request) < 5 or request[0] != FRAME_HEAD or request[-1] != FRAME_TAIL:
            raise ValueError("request is not a PD42S1 frame")
        function = request[2]
        response_length = self.RESPONSE_LENGTHS.get(function)
        if response_length is None:
            raise ValueError(f"unknown response length for function 0x{function:02X}")
        self.open()
        assert self._serial is not None
        with self._lock:
            reset = getattr(self._serial, "reset_input_buffer", None)
            if reset is not None:
                reset()
            written = self._serial.write(request)
            if written != len(request):
                raise PD42S1ProtocolError(
                    f"serial write was partial: {written}/{len(request)} bytes"
                )
            self._serial.flush()
            raw = self._read_exactly(response_length)
        return parse_response(raw, function, self.address)

    def set_position_mode(self) -> None:
        frame = self.transact(frame_set_position_mode(self.address))
        if len(frame.data) != 2 or frame.data[1] != MODE_COMMUNICATION_POSITION:
            raise PD42S1ProtocolError("driver did not echo communication position mode")

    def save_parameters(self) -> None:
        self.transact(frame_save_parameters(self.address))

    def zero_position(self) -> None:
        self.transact(frame_zero_position(self.address))

    def enable(
        self,
        enabled: bool,
        *,
        verify: bool = True,
        verification_timeout_seconds: float = 0.5,
        verification_interval_seconds: float = 0.05,
    ) -> None:
        frame = self.transact(frame_enable(enabled, self.address))
        expected = 0 if enabled else 1
        if len(frame.data) != 2 or frame.data[1] != expected:
            raise PD42S1ProtocolError("driver did not echo requested enable state")
        if not verify:
            return
        if verification_timeout_seconds <= 0:
            raise ValueError("verification_timeout_seconds must be positive")
        if verification_interval_seconds < 0:
            raise ValueError("verification_interval_seconds cannot be negative")
        if not enabled and self.stale_disabled_status_physically_verified:
            return

        deadline = time.monotonic() + verification_timeout_seconds
        last_status: SystemStatus | None = None
        last_error: Exception | None = None
        while True:
            try:
                last_status = self.read_system_status()
                last_error = None
                if last_status.enabled == enabled:
                    return
            except PD42S1Error as error:
                last_error = error

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(verification_interval_seconds, remaining))

        requested = "enabled" if enabled else "disabled"
        if last_status is not None:
            reported = "enabled" if last_status.enabled else "disabled"
            raise PD42S1ProtocolError(
                f"driver acknowledged {requested}, but 0x31 still reports {reported} "
                f"after {verification_timeout_seconds:.2f}s"
            )
        raise PD42S1ProtocolError(
            f"driver acknowledged {requested}, but 0x31 verification failed: {last_error}"
        )

    def clear_state(self) -> None:
        self.transact(frame_clear_state(self.address))

    def emergency_stop(self) -> None:
        self.transact(frame_stop_now(self.address))

    def move_absolute(self, target_count: int, speed_rpm: int, acceleration: int) -> None:
        self.transact(
            frame_absolute_position(
                target_count, speed_rpm, acceleration, self.address
            )
        )

    def read_position(self) -> int:
        return decode_position(self.transact(frame_read_position(self.address)))

    def read_system_status(self) -> SystemStatus:
        return decode_system_status(self.transact(frame_read_system(self.address)))


class MotorWorker:
    """Own the UART and coalesce high-rate control updates to the newest target."""

    def __init__(
        self,
        connection: PD42S1Connection,
        min_count: int,
        max_count: int,
        speed_rpm: int = 10,
        acceleration: int = 10,
        command_hz: float = 40.0,
        status_hz: float = 2.0,
        max_errors: int = 1,
    ) -> None:
        if min_count >= max_count:
            raise ValueError("motor soft limits are invalid")
        self.connection = connection
        self.min_count = min_count
        self.max_count = max_count
        self.speed_rpm = speed_rpm
        self.acceleration = acceleration
        self.command_period = 1.0 / command_hz
        self.status_period = 1.0 / status_hz
        self.max_errors = max_errors
        self._condition = threading.Condition()
        self._snapshot = MotorSnapshot()
        self._desired_armed = False
        self._desired_target: int | None = None
        self._estop_reason: str | None = None
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="pd42s1-uart", daemon=True)
        self._thread.start()

    def snapshot(self) -> MotorSnapshot:
        with self._condition:
            return self._snapshot

    def set_armed(self, armed: bool) -> None:
        with self._condition:
            if self._snapshot.estop and armed:
                raise PD42S1Error("cannot arm a latched emergency stop")
            self._desired_armed = armed
            self._condition.notify_all()

    def set_target(self, target_count: int) -> None:
        if not self.min_count <= target_count <= self.max_count:
            raise PD42S1Error(
                f"motor target {target_count} exceeds [{self.min_count}, {self.max_count}]"
            )
        with self._condition:
            self._desired_target = int(target_count)
            self._condition.notify_all()

    def emergency_stop(self, reason: str) -> None:
        with self._condition:
            self._estop_reason = reason
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.connection.close()

    def return_zero_and_disable(
        self, zero_count: int, tolerance_counts: int, timeout: float
    ) -> bool:
        self.set_target(zero_count)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            if (
                snapshot.actual_count is not None
                and abs(snapshot.actual_count - zero_count) <= tolerance_counts
                and snapshot.status is not None
                and snapshot.status.arrived
                and abs(snapshot.status.actual_position - zero_count)
                <= tolerance_counts
            ):
                self.set_armed(False)
                while time.monotonic() < deadline:
                    if not self.snapshot().armed:
                        return True
                    time.sleep(0.01)
                return False
            if snapshot.fault or snapshot.estop:
                return False
            time.sleep(0.02)
        return False

    def _publish(self, **changes: object) -> None:
        with self._condition:
            self._snapshot = replace(
                self._snapshot, timestamp_ns=time.time_ns(), **changes
            )
            self._condition.notify_all()

    def _issue_latched_estop(self, reason: str) -> None:
        stop_error: str | None = None
        try:
            self.connection.emergency_stop()
        except Exception as error:
            stop_error = str(error)
        fault = reason if stop_error is None else f"{reason}; FC failed: {stop_error}"
        self._publish(armed=False, estop=True, fault=fault, connected=True)

    def _run(self) -> None:
        sent_target: int | None = None
        sent_armed: bool | None = None
        last_status = 0.0
        errors = 0
        try:
            self.connection.open()
            self._publish(connected=True)
            while True:
                tick_start = time.monotonic()
                with self._condition:
                    if self._stop:
                        break
                    desired_armed = self._desired_armed
                    desired_target = self._desired_target
                    estop_reason = self._estop_reason
                if estop_reason is not None:
                    self._issue_latched_estop(estop_reason)
                    break
                try:
                    if sent_armed is None or desired_armed != sent_armed:
                        self.connection.enable(desired_armed)
                        sent_armed = desired_armed
                        self._publish(armed=desired_armed)
                    if desired_armed and desired_target is not None and desired_target != sent_target:
                        self.connection.move_absolute(
                            desired_target, self.speed_rpm, self.acceleration
                        )
                        sent_target = desired_target
                        self._publish(target_count=desired_target)
                    actual = self.connection.read_position()
                    self._publish(actual_count=actual, connected=True)
                    if not self.min_count <= actual <= self.max_count:
                        self._issue_latched_estop(
                            f"actual position {actual} is outside soft limits"
                        )
                        break
                    if tick_start - last_status >= self.status_period:
                        status = self.connection.read_system_status()
                        last_status = tick_start
                        self._publish(status=status, actual_count=status.actual_position)
                        if status.stalled:
                            self._issue_latched_estop("driver reported a stall")
                            break
                    errors = 0
                    self._publish(consecutive_errors=0, fault=None)
                except Exception as error:
                    errors += 1
                    self._publish(consecutive_errors=errors, fault=str(error))
                    if errors >= self.max_errors:
                        self._issue_latched_estop(str(error))
                        break
                elapsed = time.monotonic() - tick_start
                with self._condition:
                    if not self._stop:
                        self._condition.wait(max(0.0, self.command_period - elapsed))
        except Exception as error:
            self._publish(connected=False, fault=str(error), consecutive_errors=errors + 1)
        finally:
            self.connection.close()
