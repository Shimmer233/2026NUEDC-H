from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.calibration import fit_pipe_centerline_calibration  # noqa: E402
from steel_ball.camera_controls import apply_saved_camera_controls  # noqa: E402
from steel_ball.vision import draw_pipe_ruler_overlay  # noqa: E402


CORNER_LABELS = ("LT", "LB", "RB", "RT")
CORNER_NAMES = ("left-top", "left-bottom", "right-bottom", "right-top")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def stable_camera_frame(
    source_value: int | str,
    width: int,
    height: int,
    fps: float,
    fourcc: str,
    controls: dict,
) -> np.ndarray:
    source = int(source_value) if str(source_value).isdigit() else str(source_value)
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    capture = cv2.VideoCapture(source, backend)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"unable to open camera {source_value}")

    try:
        if sys.platform != "win32" and controls:
            control_device = (
                str(source)
                if str(source).startswith("/dev/")
                else f"/dev/video{source}"
            )
            try:
                applied = apply_saved_camera_controls(control_device, controls)
            except RuntimeError as error:
                print(f"Camera control warning: {error}", file=sys.stderr)
            else:
                if applied:
                    print(f"Camera controls: {applied}")

        frame = None
        for _ in range(30):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame read failed")
        assert frame is not None
        return frame
    finally:
        capture.release()


class PipeCornerCollector:
    def __init__(self, image: np.ndarray, length_cm: float) -> None:
        self.image = image
        self.length_cm = length_cm
        self.points: list[tuple[float, float]] = []
        self.error: str | None = None

    def mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((float(x), float(y)))
            self.error = None

    def render(self) -> np.ndarray:
        canvas = self.image.copy()
        for index, (x, y) in enumerate(self.points):
            point = (round(x), round(y))
            cv2.drawMarker(canvas, point, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(
                canvas,
                CORNER_LABELS[index],
                (point[0] + 7, point[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                canvas,
                CORNER_LABELS[index],
                (point[0] + 7, point[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
            )
        if len(self.points) >= 2:
            polyline = np.asarray(self.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [polyline], len(self.points) == 4, (0, 255, 255), 1)
        if len(self.points) == 4:
            try:
                calibration = fit_pipe_centerline_calibration(self.points, self.length_cm)
            except ValueError as error:
                self.error = str(error)
            else:
                draw_pipe_ruler_overlay(canvas, calibration)
        else:
            next_name = CORNER_NAMES[len(self.points)]
            text = f"Click {next_name} corner ({len(self.points) + 1}/4)"
            cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 3)
            cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1)
        if self.error:
            cv2.putText(canvas, self.error, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
            cv2.putText(canvas, self.error, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 1)
        return canvas


def write_report(path: Path, calibration, corners: list[tuple[float, float]]) -> None:
    pixel_length = float(calibration.length_cm / calibration.cm_per_px)
    payload = {
        "kind": calibration.kind,
        "length_cm": calibration.length_cm,
        "pixel_length": pixel_length,
        "cm_per_px": calibration.cm_per_px,
        "origin_px": calibration.origin_px.tolist(),
        "axis": calibration.axis.tolist(),
        "corners_px": [{"x_px": x, "y_px": y} for x, y in corners],
        "left_endpoint_cm": -float(calibration.length_cm) * 0.5,
        "right_endpoint_cm": float(calibration.length_cm) * 0.5,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate a 25 cm pipe centerline ruler.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "board.yaml")
    parser.add_argument("--image", type=Path, help="Calibration still image; camera is used if omitted.")
    parser.add_argument("--camera", type=int, help="Camera index used when --device is omitted.")
    parser.add_argument("--device", help="Linux V4L2 device path, for example /dev/video2")
    parser.add_argument("--width", type=int, help="Camera frame width.")
    parser.add_argument("--height", type=int, help="Camera frame height.")
    parser.add_argument("--fps", type=float, help="Camera frame rate.")
    parser.add_argument("--length-cm", type=float, default=25.0)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "config" / "track_calibration.yaml"
    )
    args = parser.parse_args()

    config = read_config(args.config)
    camera_config = config.get("camera", {})
    source_value = (
        args.device
        if args.device is not None
        else args.camera
        if args.camera is not None
        else camera_config.get("device", camera_config.get("index", 0))
    )
    width = int(args.width if args.width is not None else camera_config.get("width", 1280))
    height = int(args.height if args.height is not None else camera_config.get("height", 720))
    fps = float(args.fps if args.fps is not None else camera_config.get("fps", 60.0))
    fourcc = str(camera_config.get("fourcc", "MJPG"))
    controls = dict(camera_config.get("controls", {}))

    if args.image:
        image = cv2.imread(str(project_path(args.image)))
        if image is None:
            raise RuntimeError(f"unable to read {args.image}")
    else:
        image = stable_camera_frame(source_value, width, height, fps, fourcc, controls)

    collector = PipeCornerCollector(image, args.length_cm)
    window = "Pipe ruler calibration"
    print("Click pipe corners in this order: left-top, left-bottom, right-bottom, right-top.")
    print("Backspace undoes; Enter saves; Esc cancels.")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, collector.mouse)
    while True:
        cv2.imshow(window, collector.render())
        key = cv2.waitKey(20) & 0xFF
        if key in (8, 127) and collector.points:
            collector.points.pop()
            collector.error = None
        elif key in (13, 32) and len(collector.points) == 4:
            try:
                fit_pipe_centerline_calibration(collector.points, args.length_cm)
            except ValueError as error:
                collector.error = str(error)
                print(f"Invalid calibration points: {error}", file=sys.stderr)
            else:
                break
        elif key == 27:
            cv2.destroyAllWindows()
            return 1
    cv2.destroyAllWindows()

    calibration = fit_pipe_centerline_calibration(collector.points, args.length_cm)
    output_path = project_path(args.output)
    calibration.save(output_path)

    reference = image.copy()
    draw_pipe_ruler_overlay(reference, calibration)
    reference_path = output_path.with_suffix(".reference.jpg")
    report_path = output_path.with_suffix(".report.json")
    cv2.imwrite(str(reference_path), reference)
    write_report(report_path, calibration, collector.points)
    print(f"Calibration: {output_path}")
    print(f"Reference: {reference_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
