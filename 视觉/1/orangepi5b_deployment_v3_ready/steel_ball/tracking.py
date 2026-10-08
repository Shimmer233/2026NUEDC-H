from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackEstimate:
    available: bool
    detected: bool
    predicted: bool
    position_cm: float | None
    velocity_cm_s: float | None
    missing_frames: int


class PositionVelocityKalman:
    def __init__(
        self,
        process_acceleration_std: float = 40.0,
        measurement_std: float = 0.12,
        max_prediction_frames: int = 3,
    ) -> None:
        self.process_acceleration_std = process_acceleration_std
        self.measurement_variance = measurement_std**2
        self.max_prediction_frames = max_prediction_frames
        self.state: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.last_timestamp: float | None = None
        self.missing_frames = 0

    def reset(self) -> None:
        self.state = None
        self.covariance = None
        self.last_timestamp = None
        self.missing_frames = 0

    def _predict(self, dt: float) -> None:
        assert self.state is not None and self.covariance is not None
        transition = np.asarray([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        noise_vector = np.asarray([[0.5 * dt * dt], [dt]], dtype=np.float64)
        process_noise = (
            noise_vector @ noise_vector.T * self.process_acceleration_std**2
        )
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise

    def step(self, measurement_cm: float | None, timestamp_seconds: float) -> TrackEstimate:
        if self.state is None:
            self.last_timestamp = timestamp_seconds
            if measurement_cm is None:
                return TrackEstimate(False, False, False, None, None, 0)
            self.state = np.asarray([measurement_cm, 0.0], dtype=np.float64)
            self.covariance = np.diag([self.measurement_variance, 100.0])
            self.missing_frames = 0
            return TrackEstimate(True, True, False, measurement_cm, 0.0, 0)

        assert self.last_timestamp is not None and self.covariance is not None
        dt = float(np.clip(timestamp_seconds - self.last_timestamp, 1e-4, 0.25))
        self.last_timestamp = timestamp_seconds
        self._predict(dt)
        if measurement_cm is not None:
            observation = np.asarray([1.0, 0.0], dtype=np.float64)
            innovation = measurement_cm - float(observation @ self.state)
            innovation_variance = float(
                observation @ self.covariance @ observation.T + self.measurement_variance
            )
            gain = self.covariance @ observation / innovation_variance
            self.state += gain * innovation
            self.covariance = (
                np.eye(2, dtype=np.float64) - np.outer(gain, observation)
            ) @ self.covariance
            self.missing_frames = 0
            return TrackEstimate(
                True, True, False, float(self.state[0]), float(self.state[1]), 0
            )

        self.missing_frames += 1
        if self.missing_frames <= self.max_prediction_frames:
            return TrackEstimate(
                True,
                False,
                True,
                float(self.state[0]),
                float(self.state[1]),
                self.missing_frames,
            )
        self.reset()
        return TrackEstimate(False, False, False, None, None, 0)
