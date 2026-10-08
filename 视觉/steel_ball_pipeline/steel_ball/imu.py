from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ImuSample:
    timestamp_ns: int
    acceleration_m_s2: tuple[float, float, float]
    angular_velocity_deg_s: tuple[float, float, float]


class ImuAdapter(Protocol):
    """Reserved interface for the future /dev/ttyS1 protocol implementation."""

    def latest(self) -> ImuSample | None: ...

    def close(self) -> None: ...


class DisabledImuAdapter:
    def latest(self) -> ImuSample | None:
        return None

    def close(self) -> None:
        return None
