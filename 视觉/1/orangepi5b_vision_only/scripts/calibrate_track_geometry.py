from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.detector import TrackGeometry  # noqa: E402


class FourPointCollector:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.points: list[tuple[float, float]] = []

    def mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((float(x), float(y)))

    def render(self) -> np.ndarray:
        canvas = self.image.copy()
        for index, point in enumerate(self.points):
            location = (round(point[0]), round(point[1]))
            cv2.drawMarker(canvas, location, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(
                canvas,
                str(index + 1),
                (location[0] + 7, location[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
        return canvas


def capture_frame(config: dict, device: str) -> np.ndarray:
    camera = config["camera"]
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(camera["fourcc"])))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera["width"]))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera["height"]))
    capture.set(cv2.CAP_PROP_FPS, float(camera["fps"]))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open {device}")
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
    parser = argparse.ArgumentParser(description="Calibrate the four track ROI corners.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "vision.yaml")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "config" / "track_geometry.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    image = cv2.imread(str(args.image)) if args.image else capture_frame(config, args.device)
    if image is None:
        raise RuntimeError("unable to read calibration image")
    collector = FourPointCollector(image)
    window = "Track corners: TL, TR, BR, BL"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, collector.mouse)
    while True:
        cv2.imshow(window, collector.render())
        key = cv2.waitKey(20) & 0xFF
        if key == 8 and collector.points:
            collector.points.pop()
        elif key in (13, 32) and len(collector.points) == 4:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            return 1
    geometry = TrackGeometry(
        np.asarray(collector.points, dtype=np.float32),
        (int(config["roi"]["output_width"]), int(config["roi"]["output_height"])),
    )
    preview = geometry.warp(image)
    cv2.imshow("Rectified track preview", preview)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if key == 27:
        return 1
    payload = {
        "version": 1,
        "source_points": [[float(x), float(y)] for x, y in collector.points],
        "output_width": geometry.output_width,
        "output_height": geometry.output_height,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n"
    )
    print(f"Track geometry: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
