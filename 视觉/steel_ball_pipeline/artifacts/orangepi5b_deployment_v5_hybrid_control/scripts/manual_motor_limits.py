from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.pd42s1 import PD42S1Connection  # noqa: E402


DEFAULT_STATE = PROJECT_ROOT / "config" / "manual_motor_limits.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ManualMotorLimits:
    address: int
    center_count: int
    max_excursion_counts: int
    captured_at: str
    up: dict[str, Any] | None = None
    down: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: Path) -> "ManualMotorLimits":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != 1:
            raise ValueError("unsupported manual motor limits file")
        max_excursion = int(payload["max_excursion_counts"])
        if max_excursion <= 0:
            raise ValueError("max_excursion_counts must be positive")
        return cls(
            address=int(payload["address"]),
            center_count=int(payload["center_count"]),
            max_excursion_counts=max_excursion,
            captured_at=str(payload["captured_at"]),
            up=payload.get("up"),
            down=payload.get("down"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "address": self.address,
            "center_count": self.center_count,
            "max_excursion_counts": self.max_excursion_counts,
            "captured_at": self.captured_at,
            "up": self.up,
            "down": self.down,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        temporary.replace(path)


def connection(args: argparse.Namespace) -> PD42S1Connection:
    # Positional arguments keep this tool compatible with both existing board builds.
    return PD42S1Connection(
        args.device,
        args.baudrate,
        args.address,
        args.timeout,
    )


def wait_for_target(
    bus: PD42S1Connection,
    target: int,
    *,
    tolerance: int,
    timeout_seconds: float,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    last_status: Any = None
    while time.monotonic() < deadline:
        last_status = bus.read_system_status()
        if last_status.stalled:
            raise RuntimeError("driver reported a stall")
        if abs(last_status.actual_position - target) <= tolerance:
            return last_status
        time.sleep(0.05)
    actual = None if last_status is None else last_status.actual_position
    raise RuntimeError(f"motor did not reach {target}; last actual count was {actual}")


def move_and_hold(
    bus: PD42S1Connection,
    target: int,
    *,
    speed_rpm: int,
    acceleration: int,
    tolerance: int,
    timeout_seconds: float,
) -> Any:
    if speed_rpm <= 0 or acceleration <= 0:
        raise ValueError("speed and acceleration must be positive")
    command_sent = False
    try:
        bus.set_position_mode()
        # Set the target before enabling so a released motor cannot jump to an old target.
        bus.move_absolute(target, speed_rpm, acceleration)
        command_sent = True
        bus.enable(True)
        return wait_for_target(
            bus,
            target,
            tolerance=tolerance,
            timeout_seconds=timeout_seconds,
        )
    except BaseException:
        if command_sent:
            try:
                bus.emergency_stop()
            except Exception as stop_error:
                print(f"WARNING: emergency stop failed: {stop_error}", file=sys.stderr)
            print(
                "Motion failed. Cut motor power with the physical switch.",
                file=sys.stderr,
            )
        raise


def require_motion_confirmation(args: argparse.Namespace) -> None:
    if not args.arm or not args.confirm_clearance:
        raise RuntimeError("motion requires --arm and --confirm-clearance")


def capture_center(args: argparse.Namespace) -> int:
    if not args.confirm_level or not args.confirm_clearance:
        raise RuntimeError(
            "capture-center requires --confirm-level and --confirm-clearance"
        )
    if args.max_excursion_counts <= 0:
        raise ValueError("max excursion must be positive")
    bus = connection(args)
    try:
        center = bus.read_position()
        status = move_and_hold(
            bus,
            center,
            speed_rpm=args.speed_rpm,
            acceleration=args.acceleration,
            tolerance=args.tolerance_counts,
            timeout_seconds=args.move_timeout,
        )
    finally:
        bus.close()
    record = ManualMotorLimits(
        address=args.address,
        center_count=center,
        max_excursion_counts=args.max_excursion_counts,
        captured_at=utc_now(),
    )
    record.save(args.state)
    print(
        json.dumps(
            {
                "center_count": center,
                "actual_count": status.actual_position,
                "holding": bool(status.enabled),
                "allowed_offset": [
                    -record.max_excursion_counts,
                    record.max_excursion_counts,
                ],
                "state_file": str(args.state),
            },
            indent=2,
        )
    )
    return 0


def move(args: argparse.Namespace) -> int:
    require_motion_confirmation(args)
    record = ManualMotorLimits.load(args.state)
    if record.address != args.address:
        raise RuntimeError("saved motor address does not match --address")
    if abs(args.offset_counts) > record.max_excursion_counts:
        raise RuntimeError(
            f"offset exceeds +/-{record.max_excursion_counts}; recapture the center "
            "with an explicitly larger exploration guard only after checking clearance"
        )
    bus = connection(args)
    try:
        current = bus.read_position()
        allowed_error = record.max_excursion_counts + args.tolerance_counts
        if abs(current - record.center_count) > allowed_error:
            raise RuntimeError(
                "current count is outside the saved center guard; do not move. "
                "Level the pipe and run capture-center again"
            )
        target = record.center_count + args.offset_counts
        status = move_and_hold(
            bus,
            target,
            speed_rpm=args.speed_rpm,
            acceleration=args.acceleration,
            tolerance=args.tolerance_counts,
            timeout_seconds=args.move_timeout,
        )
    finally:
        bus.close()
    print(
        json.dumps(
            {
                "center_count": record.center_count,
                "target_count": target,
                "actual_count": status.actual_position,
                "offset_counts": status.actual_position - record.center_count,
                "holding": bool(status.enabled),
            },
            indent=2,
        )
    )
    return 0


def record_limit(args: argparse.Namespace) -> int:
    record = ManualMotorLimits.load(args.state)
    bus = connection(args)
    try:
        current = bus.read_position()
    finally:
        bus.close()
    offset = current - record.center_count
    if abs(offset) <= args.tolerance_counts:
        raise RuntimeError("limit is too close to the saved center")
    if abs(offset) > record.max_excursion_counts + args.tolerance_counts:
        raise RuntimeError("current position is outside the exploration guard")
    item = {
        "count": current,
        "offset_counts": offset,
        "angle_deg": float(args.angle_deg),
        "recorded_at": utc_now(),
    }
    updated = replace(record, **{args.side: item})
    updated.save(args.state)
    print(json.dumps({"recorded": args.side, **item}, indent=2))
    if updated.up is not None and updated.down is not None:
        if int(updated.up["offset_counts"]) * int(updated.down["offset_counts"]) >= 0:
            print(
                "WARNING: up and down limits are on the same count side of center.",
                file=sys.stderr,
            )
        print("Both limits are recorded. Return to center before ending the test.")
    return 0


def show(args: argparse.Namespace) -> int:
    record = ManualMotorLimits.load(args.state)
    bus = connection(args)
    try:
        status = bus.read_system_status()
    finally:
        bus.close()
    payload = {
        "center_count": record.center_count,
        "max_excursion_counts": record.max_excursion_counts,
        "up": record.up,
        "down": record.down,
        "live": {
            "actual_count": status.actual_position,
            "offset_counts": status.actual_position - record.center_count,
            "enabled": status.enabled,
            "arrived": status.arrived,
            "stalled": status.stalled,
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def release(args: argparse.Namespace) -> int:
    if not args.confirm_supported:
        raise RuntimeError("release requires --confirm-supported")
    bus = connection(args)
    try:
        bus.enable(False)
    finally:
        bus.close()
    print("Motor holding torque released.")
    return 0


def add_serial_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="/dev/ttyS0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--address", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)


def add_motion_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--speed-rpm", type=int, default=10)
    parser.add_argument("--acceleration", type=int, default=10)
    parser.add_argument("--tolerance-counts", type=int, default=5)
    parser.add_argument("--move-timeout", type=float, default=5.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture the current balance position and explore motor limits."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture-center")
    add_serial_options(capture_parser)
    add_motion_options(capture_parser)
    capture_parser.add_argument("--max-excursion-counts", type=int, default=200)
    capture_parser.add_argument("--confirm-level", action="store_true")
    capture_parser.add_argument("--confirm-clearance", action="store_true")
    capture_parser.set_defaults(handler=capture_center)

    move_parser = subparsers.add_parser("move")
    add_serial_options(move_parser)
    add_motion_options(move_parser)
    move_parser.add_argument("--offset-counts", type=int, required=True)
    move_parser.add_argument("--arm", action="store_true")
    move_parser.add_argument("--confirm-clearance", action="store_true")
    move_parser.set_defaults(handler=move)

    record_parser = subparsers.add_parser("record-limit")
    add_serial_options(record_parser)
    record_parser.add_argument("--side", choices=("up", "down"), required=True)
    record_parser.add_argument("--angle-deg", type=float, required=True)
    record_parser.add_argument("--tolerance-counts", type=int, default=5)
    record_parser.set_defaults(handler=record_limit)

    show_parser = subparsers.add_parser("show")
    add_serial_options(show_parser)
    show_parser.set_defaults(handler=show)

    release_parser = subparsers.add_parser("release")
    add_serial_options(release_parser)
    release_parser.add_argument("--confirm-supported", action="store_true")
    release_parser.set_defaults(handler=release)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
