from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.calibration import fit_track_calibration  # noqa: E402


class PointCollector:
    def __init__(self, image: np.ndarray, positions: list[float]) -> None:
        self.image = image
        self.positions = positions
        self.points: list[tuple[float, float, float]] = []

    def mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < len(self.positions):
            self.points.append((float(x), float(y), self.positions[len(self.points)]))

    def render(self) -> np.ndarray:
        canvas = self.image.copy()
        for x, y, position in self.points:
            point = (round(x), round(y))
            cv2.drawMarker(canvas, point, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(
                canvas,
                f"{position:g} cm",
                (point[0] + 7, point[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                canvas,
                f"{position:g} cm",
                (point[0] + 7, point[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )
        return canvas


def camera_frame(source: int | str, width: int, height: int, fps: float) -> np.ndarray:
    capture = cv2.VideoCapture(source, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    if not capture.isOpened():
        raise RuntimeError(f"unable to open camera {source}")
    frame = None
    for _ in range(30):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError("camera read failed")
    capture.release()
    assert frame is not None
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit pixel-to-centimetre track calibration.")
    parser.add_argument("--image", type=Path, help="Calibration still image; camera is used if omitted.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", help="Linux V4L2 device path, for example /dev/video2")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument(
        "--positions",
        default="0,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25",
        help="Known positions, clicked in this order.",
    )
    parser.add_argument("--degree", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "config" / "track_calibration.yaml"
    )
    args = parser.parse_args()
    positions = [float(value) for value in args.positions.split(",")]
    if len(positions) < 10:
        raise ValueError("use at least 10 known positions for the 2 mm P95 acceptance check")
    source: int | str = args.device if args.device else args.camera
    image = (
        cv2.imread(str(args.image))
        if args.image
        else camera_frame(source, args.width, args.height, args.fps)
    )
    if image is None:
        raise RuntimeError(f"unable to read {args.image}")
    collector = PointCollector(image, positions)
    window = "Track calibration"
    print("Click each physical mark in the listed order. Backspace undoes; Enter fits; Esc cancels.")
    print("Positions:", positions)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, collector.mouse)
    while True:
        cv2.imshow(window, collector.render())
        key = cv2.waitKey(20) & 0xFF
        if key == 8 and collector.points:
            collector.points.pop()
        elif key in (13, 32) and len(collector.points) == len(positions):
            break
        elif key == 27:
            cv2.destroyAllWindows()
            return 1
    cv2.destroyAllWindows()
    calibration = fit_track_calibration(collector.points, args.degree)
    calibration.save(args.output)
    errors_mm = np.asarray(
        [
            abs(calibration.position_cm(x, y) - position) * 10
            for x, y, position in collector.points
        ]
    )
    report = {
        "point_count": len(collector.points),
        "rmse_mm": calibration.rmse_cm * 10,
        "p95_error_mm": float(np.percentile(errors_mm, 95)),
        "max_error_mm": float(errors_mm.max()),
        "passes_p95_2mm_on_fit_points": bool(np.percentile(errors_mm, 95) <= 2.0),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    reference_path = args.output.with_suffix(".reference.jpg")
    cv2.imwrite(str(reference_path), collector.render())
    print(json.dumps(report, indent=2))
    print(f"Calibration: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
