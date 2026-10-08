from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the verified PD42S1 serial timeout without replacing board settings."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "board.yaml"
    )
    parser.add_argument("--timeout", type=float, default=0.2)
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    motor = config.get("motor")
    if not isinstance(motor, dict):
        raise RuntimeError(f"motor configuration is missing from {args.config}")
    previous = motor.get("timeout_seconds")
    motor["timeout_seconds"] = float(args.timeout)
    args.config.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n"
    )
    print(f"Updated {args.config}: motor.timeout_seconds {previous} -> {args.timeout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
