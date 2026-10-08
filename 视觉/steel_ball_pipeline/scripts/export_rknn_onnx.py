from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUT_SHAPE = [1, 3, 640, 640]
EXPECTED_OUTPUT_SHAPES = [
    [1, 18, 80, 80],
    [1, 18, 40, 40],
    [1, 18, 20, 20],
]
EXPECTED_ANCHORS = [
    10.0,
    13.0,
    16.0,
    30.0,
    33.0,
    23.0,
    30.0,
    61.0,
    62.0,
    45.0,
    59.0,
    119.0,
    116.0,
    90.0,
    156.0,
    198.0,
    373.0,
    326.0,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shape(value_info: object) -> list[int]:
    return [int(dimension.dim_value) for dimension in value_info.type.tensor_type.shape.dim]


def verify_onnx_contract(path: Path, anchors_path: Path) -> dict[str, object]:
    import onnx

    model = onnx.load(str(path))
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer_names]
    input_shapes = [tensor_shape(item) for item in inputs]
    output_shapes = [tensor_shape(item) for item in model.graph.output]
    if input_shapes != [EXPECTED_INPUT_SHAPE]:
        raise RuntimeError(f"incompatible ONNX input shapes: {input_shapes}")
    if output_shapes != EXPECTED_OUTPUT_SHAPES:
        raise RuntimeError(f"incompatible ONNX output shapes: {output_shapes}")
    if not anchors_path.is_file():
        raise RuntimeError(f"RKNN anchor file is missing: {anchors_path}")
    anchors = [float(value) for value in anchors_path.read_text(encoding="ascii").split()]
    if anchors != EXPECTED_ANCHORS:
        raise RuntimeError(f"deployment anchors changed: {anchors}")
    return {
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "anchors": anchors,
        "onnx_sha256": sha256(path),
        "anchors_sha256": sha256(anchors_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a reviewed YOLOv5 checkpoint for RKNN.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "steel_ball.onnx",
    )
    args = parser.parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    vendor = PROJECT_ROOT / "vendor" / "yolov5-rkopt"
    export_script = vendor / "export.py"
    if not export_script.is_file():
        raise RuntimeError(f"clean Rockchip YOLOv5 checkout is missing: {vendor}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(export_script),
        "--rknpu",
        "--weights",
        str(args.weights.resolve()),
        "--imgsz",
        "640",
        "--include",
        "onnx",
        "--opset",
        "12",
    ]
    print(subprocess.list2cmdline(command))
    environment = os.environ.copy()
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    subprocess.run(command, cwd=args.output.parent, env=environment, check=True)
    generated = args.weights.with_suffix(".onnx")
    if not generated.is_file():
        raise RuntimeError(f"export did not create {generated}")
    if generated.resolve() != args.output.resolve():
        shutil.copy2(generated, args.output)
    anchors_file = args.output.parent / "RK_anchors.txt"
    contract = verify_onnx_contract(args.output, anchors_file)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(args.weights.resolve()),
        "onnx": str(args.output.resolve()),
        "anchors": str(anchors_file.resolve()) if anchors_file.exists() else None,
        "command": command,
        "input_size": 640,
        "postprocess_in_model": False,
        "contract_verified": True,
        "contract": contract,
    }
    args.output.with_suffix(".export.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"RKNN-ready ONNX: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
