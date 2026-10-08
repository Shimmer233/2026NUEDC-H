from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CONTROL_PATTERN = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9_]+)\s+0x[0-9a-fA-F]+\s+\([^)]+\)\s*:\s*"
    r"min=(?P<minimum>-?\d+)\s+max=(?P<maximum>-?\d+)\s+"
    r"step=(?P<step>\d+)\s+default=(?P<default>-?\d+)\s+value=(?P<value>-?\d+)"
)


@dataclass(frozen=True)
class V4L2Control:
    name: str
    minimum: int
    maximum: int
    step: int
    default: int
    value: int

    def clamp(self, value: int | float) -> int:
        bounded = min(self.maximum, max(self.minimum, int(round(value))))
        step = max(1, self.step)
        quantized = self.minimum + round((bounded - self.minimum) / step) * step
        return min(self.maximum, max(self.minimum, quantized))


def parse_v4l2_controls(output: str) -> dict[str, V4L2Control]:
    controls: dict[str, V4L2Control] = {}
    for line in output.splitlines():
        match = CONTROL_PATTERN.match(line)
        if not match:
            continue
        values = match.groupdict()
        control = V4L2Control(
            name=values["name"],
            minimum=int(values["minimum"]),
            maximum=int(values["maximum"]),
            step=int(values["step"]),
            default=int(values["default"]),
            value=int(values["value"]),
        )
        controls[control.name] = control
    return controls


def list_v4l2_controls(device: str | Path) -> dict[str, V4L2Control]:
    result = subprocess.run(
        ["v4l2-ctl", "--device", str(device), "--list-ctrls-menus"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"unable to query V4L2 controls for {device}: {detail}")
    return parse_v4l2_controls(result.stdout)


def set_v4l2_controls(
    device: str | Path,
    requested: Mapping[str, int | float],
    available: Mapping[str, V4L2Control] | None = None,
) -> dict[str, int]:
    controls = dict(available) if available is not None else list_v4l2_controls(device)
    applied = {
        name: controls[name].clamp(value)
        for name, value in requested.items()
        if name in controls
    }
    if not applied:
        return {}
    assignments = ",".join(f"{name}={value}" for name, value in applied.items())
    result = subprocess.run(
        ["v4l2-ctl", "--device", str(device), f"--set-ctrl={assignments}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"unable to set V4L2 controls on {device}: {detail}")
    return applied


def safe_exposure_absolute_max(control: V4L2Control, fps: float) -> int:
    """Return a 90%-frame-period UVC exposure limit (100 us units)."""
    frame_period_units = 10_000.0 / max(1.0, fps)
    return control.clamp(min(control.maximum, frame_period_units * 0.9))


def apply_saved_camera_controls(
    device: str | Path, requested: Mapping[str, int | float]
) -> dict[str, int]:
    if not requested:
        return {}
    available = list_v4l2_controls(device)
    applied: dict[str, int] = {}
    for names in (
        ("power_line_frequency", "exposure_auto"),
        ("exposure_absolute", "gain", "brightness", "gamma"),
    ):
        values = {name: requested[name] for name in names if name in requested}
        applied.update(set_v4l2_controls(device, values, available))
    return applied
