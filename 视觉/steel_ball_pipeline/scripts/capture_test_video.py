from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2


def fourcc_text(value: int) -> str:
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a fixed-mode high-frame-rate test video.")
    parser.add_argument("--kind", choices=("sample", "normal", "fast", "empty"), required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--min-measured-fps", type=float, default=58.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\University\NUEDC\datasets\steel_ball_v2\test_sources"),
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if len(args.fourcc) != 4:
        raise ValueError("--fourcc must contain exactly four characters")
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    capture = cv2.VideoCapture(args.camera, backend)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"unable to open camera {args.camera}")
    actual = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps_reported": float(capture.get(cv2.CAP_PROP_FPS)),
        "fourcc_reported": fourcc_text(int(capture.get(cv2.CAP_PROP_FOURCC))),
    }
    if actual["width"] != args.width or actual["height"] != args.height:
        capture.release()
        raise RuntimeError(
            f"camera did not accept {args.width}x{args.height}: {actual}"
        )
    for _ in range(10):
        capture.read()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.kind}_{datetime.now():%Y%m%d_%H%M%S}"
    video_path = args.output_dir / f"{stem}.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"unable to create {video_path}")
    started = time.monotonic()
    frame_count = 0
    interrupted = False
    try:
        while time.monotonic() - started < args.seconds:
            ok, frame = capture.read()
            if not ok:
                continue
            writer.write(frame)
            frame_count += 1
            if not args.headless:
                cv2.imshow(f"Test capture: {args.kind}", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    interrupted = True
                    break
    finally:
        elapsed = time.monotonic() - started
        writer.release()
        capture.release()
        cv2.destroyAllWindows()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": args.kind,
        "video": str(video_path.resolve()),
        "camera_index": args.camera,
        "requested": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "fourcc": args.fourcc,
        },
        "actual": actual,
        "frames": frame_count,
        "elapsed_seconds": elapsed,
        "measured_capture_fps": frame_count / elapsed if elapsed else 0.0,
        "minimum_required_fps": args.min_measured_fps,
        "fps_passed": bool(elapsed and frame_count / elapsed >= args.min_measured_fps),
        "interrupted": interrupted,
        "platform": platform.platform(),
    }
    video_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if not metadata["fps_passed"]:
        print(
            f"capture rejected: measured {metadata['measured_capture_fps']:.3f} FPS, "
            f"required at least {args.min_measured_fps:.3f} FPS",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
