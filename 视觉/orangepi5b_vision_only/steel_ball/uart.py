from __future__ import annotations

import os
from dataclasses import dataclass


BAUD_RATES = {
    9600: "B9600",
    19200: "B19200",
    38400: "B38400",
    57600: "B57600",
    115200: "B115200",
    230400: "B230400",
    460800: "B460800",
    921600: "B921600",
}


def format_position_velocity_packet(position_cm: float, velocity_cm_s: float) -> bytes:
    """Return one ASCII UART packet: PV,<x_cm>,<v_cm_s>."""
    return f"PV,{position_cm:+.3f},{velocity_cm_s:+.3f}\n".encode("ascii")


@dataclass
class UartPositionVelocitySender:
    device: str = "/dev/ttyS1"
    baud_rate: int = 115200

    def __post_init__(self) -> None:
        self._fd: int | None = None

    def open(self) -> "UartPositionVelocitySender":
        if self._fd is not None:
            return self
        try:
            import termios
        except ImportError as error:
            raise RuntimeError("UART output requires Linux termios support") from error

        baud_name = BAUD_RATES.get(self.baud_rate)
        if baud_name is None:
            supported = ", ".join(str(value) for value in sorted(BAUD_RATES))
            raise ValueError(f"unsupported UART baud rate {self.baud_rate}; use {supported}")
        baud = getattr(termios, baud_name, None)
        if baud is None:
            raise ValueError(f"UART baud rate {self.baud_rate} is not supported here")

        fd = os.open(self.device, os.O_WRONLY | os.O_NOCTTY)
        try:
            attrs = termios.tcgetattr(fd)
            attrs[0] = 0
            attrs[1] = 0
            attrs[2] = termios.CLOCAL | termios.CREAD | termios.CS8
            attrs[3] = 0
            attrs[4] = baud
            attrs[5] = baud
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def send(self, position_cm: float, velocity_cm_s: float) -> None:
        if self._fd is None:
            raise RuntimeError("open UART before sending")
        os.write(self._fd, format_position_velocity_packet(position_cm, velocity_cm_s))

    def close(self) -> None:
        if self._fd is None:
            return
        os.close(self._fd)
        self._fd = None

    def __enter__(self) -> "UartPositionVelocitySender":
        return self.open()

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
