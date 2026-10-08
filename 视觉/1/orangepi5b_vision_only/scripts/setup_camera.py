from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.camera_controls import (  # noqa: E402
    V4L2Control,
    apply_saved_camera_controls,
    list_v4l2_controls,
    safe_exposure_absolute_max,
    set_v4l2_controls,
)


def read_stable_frame(capture: cv2.VideoCapture, count: int = 8) -> np.ndarray:
    frame: np.ndarray | None = None
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("camera frame read failed")
    assert frame is not None
    return frame


def roi_luma(frame: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    x, y, width, height = roi
    crop = frame[y : y + height, x : x + width]
    if crop.size == 0:
        raise ValueError(f"empty ROI: {roi}")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.percentile(gray, 70))


def stepped(control: V4L2Control, value: float) -> int:
    return control.clamp(value)


def auto_tune(
    capture: cv2.VideoCapture,
    device: str,
    roi: tuple[int, int, int, int],
    controls: dict[str, V4L2Control],
    fps: float,
    target_luma: float,
) -> tuple[dict[str, int], float, np.ndarray]:
    exposure_control = controls.get("exposure_absolute")
    gain_control = controls.get("gain")
    if exposure_control is None:
        raise RuntimeError("camera does not expose the V4L2 exposure_absolute control")

    exposure_max = safe_exposure_absolute_max(exposure_control, fps)
    exposure = stepped(exposure_control, min(max(exposure_control.value, 20), exposure_max))
    gain = gain_control.value if gain_control is not None else 0
    fixed = {}
    if "power_line_frequency" in controls:
        fixed["power_line_frequency"] = 1
    if "exposure_auto" in controls:
        fixed["exposure_auto"] = 1
    set_v4l2_controls(device, fixed, controls)

    frame = read_stable_frame(capture)
    measured = roi_luma(frame, roi)
    for _ in range(10):
        values: dict[str, int] = {"exposure_absolute": exposure}
        if gain_control is not None:
            values["gain"] = gain
        set_v4l2_controls(device, values, controls)
        frame = read_stable_frame(capture)
        measured = roi_luma(frame, roi)
        if abs(measured - target_luma) <= 8:
            break
        ratio = target_luma / max(measured, 1.0)
        if measured < target_luma:
            increased = stepped(exposure_control, min(exposure_max, exposure * min(2.0, ratio)))
            if increased > exposure:
                exposure = increased
            elif gain_control is not None and gain < gain_control.maximum:
                gain_span = gain_control.maximum - gain_control.minimum
                gain = stepped(gain_control, gain + max(gain_control.step, gain_span * 0.12))
            else:
                break
        elif gain_control is not None and gain > gain_control.minimum:
            gain_span = gain_control.maximum - gain_control.minimum
            gain = stepped(gain_control, gain - max(gain_control.step, gain_span * 0.12))
        else:
            exposure = stepped(exposure_control, max(exposure_control.minimum, exposure * ratio))

    final_values: dict[str, int] = {"exposure_absolute": exposure}
    if gain_control is not None:
        final_values["gain"] = gain
    set_v4l2_controls(device, final_values, controls)
    frame = read_stable_frame(capture)
    measured = roi_luma(frame, roi)
    saved = dict(fixed)
    saved.update(final_values)
    return saved, measured, frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Select track ROI and tune a V4L2 USB camera.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "vision.yaml")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--target-luma", type=float, default=200.0)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    camera = config["camera"]
    capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(camera["fourcc"])))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera["width"]))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera["height"]))
    capture.set(cv2.CAP_PROP_FPS, float(camera["fps"]))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"unable to open {args.device}")

    try:
        applied = apply_saved_camera_controls(
            args.device, camera.get("controls", {})
        )
        if applied:
            print(f"Initial camera controls: {applied}")
        frame = read_stable_frame(capture, 20)
        print("Drag one rectangle around the complete white track, then press Enter.")
        selected = cv2.selectROI("Select complete track ROI", frame, False, False)
        cv2.destroyWindow("Select complete track ROI")
        roi = tuple(int(value) for value in selected)
        if roi[2] <= 0 or roi[3] <= 0:
            print("ROI selection cancelled; configuration was not changed.")
            return 1
        controls = list_v4l2_controls(args.device)
        saved, measured, final_frame = auto_tune(
            capture,
            args.device,
            roi,
            controls,
            float(camera["fps"]),
            args.target_luma,
        )
        x, y, width, height = roi
        preview = final_frame.copy()
        cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 0), 2)
        print(f"Tuned ROI luma={measured:.1f}; controls={saved}")
        print("Press Enter to save or Esc to cancel.")
        cv2.imshow("Camera setup preview", preview)
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32, ord("s")):
                break
            if key == 27:
                print("Configuration was not changed.")
                return 1

        config["camera"]["device"] = args.device
        config["camera"]["controls"] = saved
        config["roi"].update(
            {"x": x, "y": y, "width": width, "height": height}
        )
        config["roi"]["output_width"] = 470
        config["roi"]["output_height"] = 110
        args.config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n"
        )
        print(f"Saved camera controls and ROI to {args.config}")
        time.sleep(0.2)
        return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
