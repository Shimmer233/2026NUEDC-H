from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


LINUX_BOOT_ID = Path("/proc/sys/kernel/random/boot_id")


@dataclass(frozen=True)
class SessionZeroMarker:
    version: int
    address: int
    zero_count: int
    tolerance_counts: int
    boot_id: str
    created_at: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8", newline="\n"
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SessionZeroMarker":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != 1:
            raise RuntimeError("unsupported motor session-zero marker version")
        return cls(
            version=1,
            address=int(payload["address"]),
            zero_count=int(payload["zero_count"]),
            tolerance_counts=int(payload["tolerance_counts"]),
            boot_id=str(payload["boot_id"]),
            created_at=str(payload["created_at"]),
        )


def read_boot_id(path: Path = LINUX_BOOT_ID) -> str:
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise RuntimeError(f"cannot read Linux boot ID from {path}: {error}") from error
    if not value:
        raise RuntimeError(f"Linux boot ID is empty: {path}")
    return value


def create_session_zero_marker(
    path: Path,
    *,
    address: int,
    zero_count: int,
    tolerance_counts: int,
    boot_id: str | None = None,
) -> SessionZeroMarker:
    if tolerance_counts < 0:
        raise ValueError("session-zero tolerance cannot be negative")
    marker = SessionZeroMarker(
        version=1,
        address=int(address),
        zero_count=int(zero_count),
        tolerance_counts=int(tolerance_counts),
        boot_id=boot_id or read_boot_id(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    marker.save(path)
    return marker


def validate_session_zero_marker(
    path: Path,
    *,
    address: int,
    zero_count: int,
    tolerance_counts: int,
    min_count: int,
    max_count: int,
    current_count: int,
    boot_id: str | None = None,
) -> SessionZeroMarker:
    if not path.is_file():
        raise RuntimeError(
            "motor session zero is missing; level the pipe and run session-zero"
        )
    marker = SessionZeroMarker.load(path)
    expected_boot_id = boot_id or read_boot_id()
    if marker.boot_id != expected_boot_id:
        raise RuntimeError(
            "motor session zero belongs to a previous system boot; run session-zero again"
        )
    if marker.address != int(address):
        raise RuntimeError("motor session-zero address does not match the driver")
    tolerance = min(int(tolerance_counts), marker.tolerance_counts)
    if abs(marker.zero_count - int(zero_count)) > tolerance:
        raise RuntimeError("motor calibration changed after session-zero was recorded")
    if not int(min_count) <= int(current_count) <= int(max_count):
        raise RuntimeError(
            f"motor count {current_count} is outside session soft limits "
            f"[{min_count}, {max_count}]"
        )
    if abs(int(current_count) - marker.zero_count) > tolerance:
        raise RuntimeError(
            f"motor count changed by {abs(int(current_count) - marker.zero_count)} "
            f"after session-zero; limit is {tolerance}"
        )
    return marker


def invalidate_session_zero_marker(path: Path) -> None:
    path.unlink(missing_ok=True)
