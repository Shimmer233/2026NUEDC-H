from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.control import fit_state_feedback_gains  # noqa: E402


def load_samples(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    positions: list[float] = []
    angles: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row.get("position_cm") or not row.get("theta_cmd_deg"):
                continue
            timestamps.append(float(row["timestamp_ns"]) / 1e9)
            positions.append(float(row["position_cm"]))
            angles.append(float(row["theta_cmd_deg"]))
    if len(timestamps) < 80:
        raise ValueError("identification log needs at least 80 valid samples")
    time_values = np.asarray(timestamps, dtype=np.float64)
    position_values = np.asarray(positions, dtype=np.float64)
    angle_values = np.asarray(angles, dtype=np.float64)
    order = np.argsort(time_values)
    time_values = time_values[order]
    position_values = position_values[order]
    angle_values = angle_values[order]
    unique = np.concatenate(([True], np.diff(time_values) > 1e-5))
    return time_values[unique], position_values[unique], angle_values[unique]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit safe ball-position controller gains.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "runs" / "board" / "control_telemetry.csv",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "board.yaml"
    )
    parser.add_argument("--settling-seconds", type=float, default=3.0)
    parser.add_argument("--damping-ratio", type=float, default=0.9)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "runs" / "board" / "control_identification.json",
    )
    parser.add_argument("--min-r-squared", type=float, default=0.20)
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Write identified gains to board.yaml after the fit passes quality gates.",
    )
    args = parser.parse_args()
    timestamps, positions, angles = load_samples(args.input)
    if np.ptp(angles) < 0.05:
        raise ValueError("identification angle does not contain both tilt pulses")
    velocity = np.gradient(positions, timestamps)
    acceleration = np.gradient(velocity, timestamps)
    finite = np.isfinite(acceleration) & np.isfinite(velocity) & np.isfinite(angles)
    finite[:3] = False
    finite[-3:] = False
    design = np.column_stack((angles[finite], -velocity[finite], np.ones(finite.sum())))
    coefficients, _, _, _ = np.linalg.lstsq(design, acceleration[finite], rcond=None)
    plant_gain, velocity_damping, bias = (float(value) for value in coefficients)
    predicted = design @ coefficients
    residual = acceleration[finite] - predicted
    rms = float(np.sqrt(np.mean(residual**2)))
    centered = acceleration[finite] - np.mean(acceleration[finite])
    total_variance = float(np.sum(centered**2))
    r_squared = (
        1.0 - float(np.sum(residual**2)) / total_variance
        if total_variance > 1e-12
        else float("-inf")
    )
    quality_errors: list[str] = []
    if abs(plant_gain) < 0.05:
        quality_errors.append("identified plant gain is too close to zero")
    if velocity_damping < -0.2:
        quality_errors.append("identified velocity damping is strongly negative")
    if r_squared < args.min_r_squared:
        quality_errors.append(
            f"fit R^2 {r_squared:.3f} is below {args.min_r_squared:.3f}"
        )
    gains = fit_state_feedback_gains(
        plant_gain,
        velocity_damping,
        args.settling_seconds,
        args.damping_ratio,
    )
    proposed_gains = {
        "kp_deg_per_cm": gains.kp_deg_per_cm,
        "kv_deg_per_cm_s": gains.kv_deg_per_cm_s,
        "ki_deg_per_cm_s": gains.ki_deg_per_cm_s,
        "identified": False,
    }
    report = {
        "samples": int(finite.sum()),
        "plant_gain_cm_s2_per_deg": plant_gain,
        "velocity_damping_per_s": velocity_damping,
        "bias_cm_s2": bias,
        "fit_rms_cm_s2": rms,
        "fit_r_squared": r_squared,
        "quality_passed": not quality_errors,
        "quality_errors": quality_errors,
        "settling_seconds": args.settling_seconds,
        "damping_ratio": args.damping_ratio,
        "proposed_gains": proposed_gains,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if quality_errors:
        raise RuntimeError("identification quality gate failed; board config was not changed")
    if not args.confirm_write:
        print("Fit passed. Review the report, then rerun with --confirm-write to enable it.")
        return 0
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["control"]["gains"] = {**proposed_gains, "identified": True}
    args.config.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n"
    )
    print(f"Updated and enabled controller gains: {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
