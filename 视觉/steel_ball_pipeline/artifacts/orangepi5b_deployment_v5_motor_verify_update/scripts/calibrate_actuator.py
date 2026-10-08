from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.control import (  # noqa: E402
    ActuatorCalibration,
    fit_actuator_calibration,
)
from steel_ball.pd42s1 import PD42S1Connection  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "config" / "actuator_calibration.yaml"


def connection(args: argparse.Namespace) -> PD42S1Connection:
    return PD42S1Connection(
        device=args.device,
        baudrate=args.baudrate,
        address=args.address,
        timeout_seconds=args.timeout,
    )


def zero(args: argparse.Namespace) -> int:
    if not args.confirm_level:
        raise RuntimeError("refusing to clear zero without --confirm-level")
    bus = connection(args)
    try:
        bus.set_position_mode()
        bus.zero_position()
        bus.save_parameters()
        time.sleep(0.1)
        count = bus.read_position()
    finally:
        bus.close()
    if not args.min_count < count < args.max_count:
        raise RuntimeError("reported zero is outside the requested soft limits")
    calibration = ActuatorCalibration(
        address=args.address,
        zero_count=count,
        counts_per_degree=0.0,
        min_count=args.min_count,
        max_count=args.max_count,
        zero_tolerance_counts=args.zero_tolerance,
        persistence_verified=False,
        points=((0.0, count),),
        calibrated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    calibration.save(args.output)
    print(f"Zeroed and saved driver parameters. Current count: {count}")
    print(f"Calibration state: {args.output}")
    print("Power-cycle without moving the mechanism, then run the verify command.")
    return 0


def status(args: argparse.Namespace) -> int:
    bus = connection(args)
    try:
        position = bus.read_position()
        system = bus.read_system_status()
    finally:
        bus.close()
    print(
        json.dumps(
            {
                "position": position,
                "speed_rpm": system.speed_rpm,
                "target_position": system.target_position,
                "position_error": system.position_error,
                "enabled": system.enabled,
                "arrived": system.arrived,
                "stalled": system.stalled,
                "bus_voltage_v": system.bus_voltage_v,
            },
            indent=2,
        )
    )
    return 0


def disable(args: argparse.Namespace) -> int:
    if not args.confirm_supported:
        raise RuntimeError(
            "refusing to release motor holding torque without --confirm-supported"
        )
    bus = connection(args)
    try:
        bus.enable(False)
        system = bus.read_system_status()
    finally:
        bus.close()
    print(
        json.dumps(
            {
                "disabled_verified": not system.enabled,
                "enabled": system.enabled,
                "phase_current_ma": system.phase_current_ma,
                "position": system.actual_position,
            },
            indent=2,
        )
    )
    return 0


def verify(args: argparse.Namespace) -> int:
    if not args.confirm_power_cycled or not args.confirm_not_moved:
        raise RuntimeError(
            "verification requires --confirm-power-cycled and --confirm-not-moved"
        )
    calibration = ActuatorCalibration.load(args.output)
    bus = connection(args)
    try:
        count = bus.read_position()
    finally:
        bus.close()
    difference = abs(count - calibration.zero_count)
    if difference > calibration.zero_tolerance_counts:
        print(
            f"FAILED: count changed by {difference}, limit is "
            f"{calibration.zero_tolerance_counts}. Balance remains blocked.",
            file=sys.stderr,
        )
        return 2
    replace(calibration, persistence_verified=True).save(args.output)
    print(f"PASSED: post-power-cycle count={count}, difference={difference}")
    print("Zero persistence is now enabled in the actuator calibration file.")
    return 0


def jog(args: argparse.Namespace) -> int:
    if not args.arm:
        raise RuntimeError("jogging requires --arm")
    calibration = ActuatorCalibration.load(args.output)
    if not calibration.persistence_verified:
        raise RuntimeError(
            "jogging is blocked until the power-cycle zero verification passes"
        )
    if not calibration.min_count <= args.target_count <= calibration.max_count:
        raise RuntimeError("target count exceeds the calibrated soft limits")
    bus = connection(args)
    completed_safely = False
    try:
        bus.enable(True)
        bus.move_absolute(args.target_count, args.speed_rpm, args.acceleration)
        deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < deadline:
            current = bus.read_position()
            print(f"count={current}", end="\r", flush=True)
            time.sleep(0.1)
        print()
        bus.move_absolute(calibration.zero_count, args.speed_rpm, args.acceleration)
        deadline = time.monotonic() + args.return_timeout
        while time.monotonic() < deadline:
            current = bus.read_position()
            if abs(current - calibration.zero_count) <= calibration.zero_tolerance_counts:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("motor did not return to zero; use the physical power switch")
        bus.enable(False)
        completed_safely = True
    except BaseException:
        try:
            bus.emergency_stop()
        except Exception as stop_error:
            print(f"WARNING: software emergency stop failed: {stop_error}", file=sys.stderr)
        print(
            "ESTOP requested. Cut motor power with the physical switch and inspect the mechanism.",
            file=sys.stderr,
        )
        raise
    finally:
        bus.close()
    if not completed_safely:
        return 2
    print("Returned to zero and disabled the motor.")
    return 0


def parse_point(value: str) -> tuple[float, int]:
    try:
        angle, count = value.split(":", 1)
        return float(angle), int(count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("point must be ANGLE_DEG:COUNT") from error


def fit(args: argparse.Namespace) -> int:
    previous = ActuatorCalibration.load(args.output) if args.output.is_file() else None
    calibration = fit_actuator_calibration(
        args.point,
        args.min_count,
        args.max_count,
        address=args.address,
        zero_tolerance_counts=args.zero_tolerance,
        persistence_verified=bool(previous and previous.persistence_verified),
    )
    if (
        previous is not None
        and previous.persistence_verified
        and abs(calibration.zero_count - previous.zero_count)
        > previous.zero_tolerance_counts
    ):
        raise RuntimeError(
            "fitted zero moved outside the verified tolerance; calibration was not saved"
        )
    calibration.save(args.output)
    print(
        json.dumps(
            {
                "zero_count": calibration.zero_count,
                "counts_per_degree": calibration.counts_per_degree,
                "min_count": calibration.min_count,
                "max_count": calibration.max_count,
                "persistence_verified": calibration.persistence_verified,
            },
            indent=2,
        )
    )
    return 0


def add_serial_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="/dev/ttyS0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--address", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)


def main() -> int:
    parser = argparse.ArgumentParser(description="One-time PD42S1 zero and angle calibration.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    zero_parser = subparsers.add_parser("zero")
    add_serial_options(zero_parser)
    zero_parser.add_argument("--confirm-level", action="store_true")
    zero_parser.add_argument("--min-count", type=int, required=True)
    zero_parser.add_argument("--max-count", type=int, required=True)
    zero_parser.add_argument("--zero-tolerance", type=int, default=20)
    zero_parser.set_defaults(handler=zero)

    status_parser = subparsers.add_parser("status")
    add_serial_options(status_parser)
    status_parser.set_defaults(handler=status)

    disable_parser = subparsers.add_parser("disable")
    add_serial_options(disable_parser)
    disable_parser.add_argument("--confirm-supported", action="store_true")
    disable_parser.set_defaults(handler=disable)

    verify_parser = subparsers.add_parser("verify")
    add_serial_options(verify_parser)
    verify_parser.add_argument("--confirm-power-cycled", action="store_true")
    verify_parser.add_argument("--confirm-not-moved", action="store_true")
    verify_parser.set_defaults(handler=verify)

    jog_parser = subparsers.add_parser("jog")
    add_serial_options(jog_parser)
    jog_parser.add_argument("--target-count", type=int, required=True)
    jog_parser.add_argument("--speed-rpm", type=int, default=10)
    jog_parser.add_argument("--acceleration", type=int, default=10)
    jog_parser.add_argument("--hold-seconds", type=float, default=5.0)
    jog_parser.add_argument("--return-timeout", type=float, default=3.0)
    jog_parser.add_argument("--arm", action="store_true")
    jog_parser.set_defaults(handler=jog)

    fit_parser = subparsers.add_parser("fit")
    add_serial_options(fit_parser)
    fit_parser.add_argument("--point", type=parse_point, action="append", required=True)
    fit_parser.add_argument("--min-count", type=int, required=True)
    fit_parser.add_argument("--max-count", type=int, required=True)
    fit_parser.add_argument("--zero-tolerance", type=int, default=20)
    fit_parser.set_defaults(handler=fit)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
