from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.dataset import verify_split_review  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the independent test split.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\University\NUEDC\datasets\steel_ball_v2"),
    )
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    reviewed, reason = verify_split_review(args.dataset, "test")
    if not reviewed:
        raise RuntimeError(f"test evaluation blocked: {reason}")
    yolov5 = PROJECT_ROOT / "vendor" / "yolov5-rkopt"
    run_name = f"test_{args.weights.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    command = [
        sys.executable,
        str(yolov5 / "val.py"),
        "--data",
        str(args.dataset / "steel_ball.yaml"),
        "--weights",
        str(args.weights),
        "--batch-size",
        "16",
        "--imgsz",
        "640",
        "--task",
        "test",
        "--device",
        args.device,
        "--save-txt",
        "--save-conf",
        "--project",
        str(PROJECT_ROOT / "runs" / "eval"),
        "--name",
        run_name,
    ]
    print(subprocess.list2cmdline(command))
    subprocess.run(command, cwd=yolov5, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
