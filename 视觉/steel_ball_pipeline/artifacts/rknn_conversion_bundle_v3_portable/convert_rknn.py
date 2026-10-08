from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def check_return(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError(f"RKNN {operation} failed with code {code}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert RKNN-ready YOLOv5 ONNX for RK3588.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--mode", choices=("fp16", "int8"), required=True)
    parser.add_argument("--dataset-list", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if args.mode == "int8" and (args.dataset_list is None or not args.dataset_list.is_file()):
        raise RuntimeError("INT8 conversion requires --dataset-list from the approved train split")
    try:
        toolkit_version = metadata.version("rknn-toolkit2")
        from rknn.api import RKNN
    except (metadata.PackageNotFoundError, ImportError) as error:
        raise RuntimeError(
            "RKNN-Toolkit2 is unavailable. Run this script in the pinned Linux conversion environment."
        ) from error
    if toolkit_version != "2.3.2":
        raise RuntimeError(f"expected RKNN-Toolkit2 2.3.2, found {toolkit_version}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        check_return(
            rknn.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform="rk3588",
            ),
            "config",
        )
        check_return(rknn.load_onnx(model=str(args.onnx)), "load_onnx")
        check_return(
            rknn.build(
                do_quantization=args.mode == "int8",
                dataset=str(args.dataset_list) if args.dataset_list else None,
            ),
            "build",
        )
        check_return(rknn.export_rknn(str(args.output)), "export_rknn")
    finally:
        rknn.release()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "toolkit_version": toolkit_version,
        "target_platform": "rk3588",
        "mode": args.mode,
        "onnx": str(args.onnx.resolve()),
        "dataset_list": str(args.dataset_list.resolve()) if args.dataset_list else None,
        "output": str(args.output.resolve()),
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"RKNN model: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
