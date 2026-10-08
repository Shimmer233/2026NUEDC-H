from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


class ControlState(str, Enum):
    ZERO_UNVERIFIED = "ZERO_UNVERIFIED"
    READY = "READY"
    ARMED = "ARMED"
    LOST = "LOST"
    MOTOR_FAULT = "MOTOR_FAULT"
    ESTOP = "ESTOP"


@dataclass(frozen=True)
class ControllerGains:
    kp_deg_per_cm: float
    kv_deg_per_cm_s: float
    ki_deg_per_cm_s: float
    identified: bool = False


@dataclass(frozen=True)
class ActuatorCalibration:
    address: int
    zero_count: int
    counts_per_degree: float
    min_count: int
    max_count: int
    zero_tolerance_counts: int
    persistence_verified: bool
    points: tuple[tuple[float, int], ...]
    calibrated_at: str

    @property
    def mapping_ready(self) -> bool:
        return (
            abs(self.counts_per_degree) > 1e-9
            and self.min_count < self.zero_count < self.max_count
        )

    @property
    def ready(self) -> bool:
        return self.persistence_verified and self.mapping_ready

    def angle_to_count(self, angle_deg: float) -> int:
        count = round(self.zero_count + angle_deg * self.counts_per_degree)
        if not self.min_count <= count <= self.max_count:
            raise ValueError(
                f"angle {angle_deg:.4f} deg maps outside motor soft limits"
            )
        return count

    def count_to_angle(self, count: int) -> float:
        if abs(self.counts_per_degree) <= 1e-9:
            raise ValueError("actuator angle mapping has not been fitted")
        return (count - self.zero_count) / self.counts_per_degree

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "address": self.address,
            "zero_count": self.zero_count,
            "counts_per_degree": self.counts_per_degree,
            "min_count": self.min_count,
            "max_count": self.max_count,
            "zero_tolerance_counts": self.zero_tolerance_counts,
            "persistence_verified": self.persistence_verified,
            "calibrated_at": self.calibrated_at,
            "points": [
                {"angle_deg": angle, "count": count} for angle, count in self.points
            ],
        }
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n"
        )

    @classmethod
    def load(cls, path: Path) -> "ActuatorCalibration":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != 1:
            raise ValueError("unsupported actuator calibration version")
        return cls(
            address=int(payload["address"]),
            zero_count=int(payload["zero_count"]),
            counts_per_degree=float(payload["counts_per_degree"]),
            min_count=int(payload["min_count"]),
            max_count=int(payload["max_count"]),
            zero_tolerance_counts=int(payload["zero_tolerance_counts"]),
            persistence_verified=bool(payload["persistence_verified"]),
            points=tuple(
                (float(item["angle_deg"]), int(item["count"]))
                for item in payload.get("points", [])
            ),
            calibrated_at=str(payload["calibrated_at"]),
        )


def fit_actuator_calibration(
    points: Sequence[tuple[float, int]],
    min_count: int,
    max_count: int,
    address: int = 1,
    zero_tolerance_counts: int = 20,
    persistence_verified: bool = False,
) -> ActuatorCalibration:
    if len(points) < 3:
        raise ValueError("use at least three negative/zero/positive angle points")
    values = np.asarray(points, dtype=np.float64)
    angles = values[:, 0]
    counts = values[:, 1]
    if np.ptp(angles) < 0.1:
        raise ValueError("actuator calibration angles do not span enough range")
    slope, intercept = np.polyfit(angles, counts, 1)
    if abs(slope) < 1.0:
        raise ValueError("invalid counts-per-degree fit")
    zero_count = int(round(intercept))
    if not min_count < zero_count < max_count:
        raise ValueError("fitted zero is outside the requested soft limits")
    return ActuatorCalibration(
        address=address,
        zero_count=zero_count,
        counts_per_degree=float(slope),
        min_count=int(min_count),
        max_count=int(max_count),
        zero_tolerance_counts=int(zero_tolerance_counts),
        persistence_verified=persistence_verified,
        points=tuple((float(angle), int(count)) for angle, count in points),
        calibrated_at=datetime.now(timezone.utc).isoformat(),
    )


def fit_state_feedback_gains(
    plant_gain_cm_s2_per_deg: float,
    velocity_damping_per_s: float,
    settling_time_seconds: float = 3.0,
    damping_ratio: float = 0.9,
) -> ControllerGains:
    if abs(plant_gain_cm_s2_per_deg) < 1e-6:
        raise ValueError("plant gain is too small")
    if settling_time_seconds <= 0 or not 0.2 <= damping_ratio <= 2.0:
        raise ValueError("invalid closed-loop design target")
    omega = 4.0 / (damping_ratio * settling_time_seconds)
    kp = omega**2 / plant_gain_cm_s2_per_deg
    kv = (2.0 * damping_ratio * omega - velocity_damping_per_s) / plant_gain_cm_s2_per_deg
    ki = kp * 0.08
    return ControllerGains(kp, kv, ki, identified=True)


@dataclass(frozen=True)
class BalanceOutput:
    target_cm: float
    requested_target_cm: float
    error_cm: float | None
    angle_deg: float
    target_count: int
    saturated: bool
    state: ControlState


class BalanceController:
    def __init__(
        self,
        calibration: ActuatorCalibration,
        gains: ControllerGains,
        initial_target_cm: float = 12.5,
        target_min_cm: float = 1.0,
        target_max_cm: float = 24.0,
        target_slew_cm_s: float = 5.0,
        max_angle_deg: float = 0.8,
        integral_band_cm: float = 2.0,
        integral_angle_limit_deg: float = 0.2,
    ) -> None:
        if target_min_cm >= target_max_cm:
            raise ValueError("target range is invalid")
        if max_angle_deg <= 0 or target_slew_cm_s <= 0:
            raise ValueError("controller limits must be positive")
        self.calibration = calibration
        self.gains = gains
        self.target_min_cm = target_min_cm
        self.target_max_cm = target_max_cm
        self.target_slew_cm_s = target_slew_cm_s
        self.max_angle_deg = max_angle_deg
        self.integral_band_cm = integral_band_cm
        self.integral_angle_limit_deg = integral_angle_limit_deg
        self.requested_target_cm = self._clamp_target(initial_target_cm)
        self.active_target_cm = self.requested_target_cm
        self.integral_cm_s = 0.0
        self.last_timestamp_seconds: float | None = None

    def _clamp_target(self, target_cm: float) -> float:
        if not math.isfinite(target_cm):
            raise ValueError("target position must be finite")
        return float(np.clip(target_cm, self.target_min_cm, self.target_max_cm))

    def set_target(self, target_cm: float) -> float:
        self.requested_target_cm = self._clamp_target(target_cm)
        return self.requested_target_cm

    def reset(self) -> None:
        self.integral_cm_s = 0.0
        self.last_timestamp_seconds = None

    def neutral(self, state: ControlState = ControlState.LOST) -> BalanceOutput:
        self.integral_cm_s = 0.0
        return BalanceOutput(
            target_cm=self.active_target_cm,
            requested_target_cm=self.requested_target_cm,
            error_cm=None,
            angle_deg=0.0,
            target_count=self.calibration.angle_to_count(0.0),
            saturated=False,
            state=state,
        )

    def update(
        self,
        position_cm: float | None,
        velocity_cm_s: float | None,
        timestamp_seconds: float,
        measurement_valid: bool,
        predicted: bool,
        armed: bool,
    ) -> BalanceOutput:
        if self.last_timestamp_seconds is None:
            dt = 0.0
        else:
            dt = float(np.clip(timestamp_seconds - self.last_timestamp_seconds, 0.0, 0.1))
        self.last_timestamp_seconds = timestamp_seconds
        max_target_step = self.target_slew_cm_s * dt
        target_delta = self.requested_target_cm - self.active_target_cm
        self.active_target_cm += float(
            np.clip(target_delta, -max_target_step, max_target_step)
        )
        if not measurement_valid or position_cm is None or velocity_cm_s is None:
            return self.neutral(ControlState.LOST)

        error = self.active_target_cm - position_cm
        proportional_derivative = (
            self.gains.kp_deg_per_cm * error
            - self.gains.kv_deg_per_cm_s * velocity_cm_s
        )
        if (
            not predicted
            and abs(error) <= self.integral_band_cm
            and abs(proportional_derivative) < self.max_angle_deg
            and dt > 0
        ):
            self.integral_cm_s += error * dt
            if abs(self.gains.ki_deg_per_cm_s) > 1e-12:
                integral_limit = self.integral_angle_limit_deg / abs(
                    self.gains.ki_deg_per_cm_s
                )
                self.integral_cm_s = float(
                    np.clip(self.integral_cm_s, -integral_limit, integral_limit)
                )
        raw_angle = (
            proportional_derivative
            + self.gains.ki_deg_per_cm_s * self.integral_cm_s
        )
        angle = float(np.clip(raw_angle, -self.max_angle_deg, self.max_angle_deg))
        saturated = abs(raw_angle - angle) > 1e-9
        return BalanceOutput(
            target_cm=self.active_target_cm,
            requested_target_cm=self.requested_target_cm,
            error_cm=error,
            angle_deg=angle,
            target_count=self.calibration.angle_to_count(angle),
            saturated=saturated,
            state=ControlState.ARMED if armed else ControlState.READY,
        )
