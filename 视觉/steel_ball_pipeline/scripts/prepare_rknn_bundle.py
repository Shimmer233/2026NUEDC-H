from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLKIT_COMMIT = "42aa1d426c0a9e0869b6374edba009f7208a1926"
TOOLKIT_WHEEL = (
    "rknn-toolkit2/packages/x86_64/"
    "rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)
TOOLKIT_REQUIREMENTS = "rknn-toolkit2/packages/x86_64/requirements_cp310-2.3.2.txt"
LITE_WHEEL = (
    "rknn-toolkit-lite2/packages/"
    "rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
)
ARM_TOOLKIT_WHEEL = (
    "rknn-toolkit2/packages/arm64/"
    "rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
)
ARM_TOOLKIT_REQUIREMENTS = "rknn-toolkit2/packages/arm64/arm64_requirements_cp310.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_scripts(output: Path) -> None:
    (output / "run_conversion.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "Conversion requires x86_64 Linux; current architecture: $(uname -m)" >&2
  exit 2
fi
if ! command -v python3.10 >/dev/null 2>&1; then
  echo "python3.10 is required (Ubuntu 22.04 default)." >&2
  exit 2
fi

python3.10 verify_bundle.py
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r toolkit/requirements_cp310-2.3.2.txt
python -m pip install toolkit/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

python convert_rknn.py --onnx model/best.onnx --mode fp16 --output output/steel_ball_yolov5n_fp16.rknn
python convert_rknn.py --onnx model/best.onnx --mode int8 --dataset-list dataset.txt --output output/steel_ball_yolov5n_int8.rknn
sha256sum output/*.rknn | tee output/SHA256SUMS.txt
echo "Conversion complete. Copy output/*.rknn to the Orange Pi deployment directory."
""",
        encoding="utf-8",
        newline="\n",
    )
    (output / "install_board_lite.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "RKNNLite2 installation must run on the Orange Pi (aarch64)." >&2
  exit 2
fi
python3.10 -m pip install board_wheel/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
""",
        encoding="utf-8",
        newline="\n",
    )
    (output / "run_conversion_arm64.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This script is for the Orange Pi / aarch64; current architecture: $(uname -m)" >&2
  exit 2
fi
if ! command -v python3.10 >/dev/null 2>&1; then
  echo "python3.10 is required (Ubuntu 22.04 default)." >&2
  exit 2
fi

python3.10 verify_bundle.py
python3.10 -m venv .venv-convert
source .venv-convert/bin/activate
python -m pip install --upgrade pip
python -m pip install -r toolkit_arm64/arm64_requirements_cp310.txt
python -m pip install toolkit_arm64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

python convert_rknn.py --onnx model/best.onnx --mode fp16 --output output/steel_ball_yolov5n_fp16.rknn
python convert_rknn.py --onnx model/best.onnx --mode int8 --dataset-list dataset.txt --output output/steel_ball_yolov5n_int8.rknn
sha256sum output/*.rknn | tee output/SHA256SUMS.txt
echo "Conversion complete on Orange Pi."
""",
        encoding="utf-8",
        newline="\n",
    )
    (output / "run_conversion_existing_env.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${RKNN_PYTHON:-python3.10}"
"$PYTHON_BIN" verify_bundle.py
"$PYTHON_BIN" - <<'PY'
from importlib import metadata

version = metadata.version("rknn-toolkit2")
if version != "2.3.2":
    raise SystemExit(f"expected rknn-toolkit2 2.3.2, found {version}")
from rknn.api import RKNN  # noqa: F401
print(f"using rknn-toolkit2 {version}")
PY

"$PYTHON_BIN" convert_rknn.py --onnx model/best.onnx --mode fp16 --output output/steel_ball_yolov5n_fp16.rknn
"$PYTHON_BIN" convert_rknn.py --onnx model/best.onnx --mode int8 --dataset-list dataset.txt --output output/steel_ball_yolov5n_int8.rknn
sha256sum output/*.rknn | tee output/SHA256SUMS.txt
echo "Conversion complete with the existing RKNN-Toolkit2 environment."
""",
        encoding="utf-8",
        newline="\n",
    )
    (output / "verify_bundle.py").write_text(
        """from __future__ import annotations
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
for relative, expected in manifest["portable_file_sha256"].items():
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"missing bundle file: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"hash mismatch: {relative}: {actual} != {expected}")
lines = [line.strip() for line in (root / "dataset.txt").read_text().splitlines() if line.strip()]
if len(lines) != manifest["calibration"]["images"]:
    raise SystemExit("dataset.txt image count mismatch")
if any(any(character.isspace() for character in line) for line in lines):
    raise SystemExit("dataset.txt paths must not contain whitespace")
if any(not (root / line).is_file() for line in lines):
    raise SystemExit("dataset.txt contains a missing image")
print(f"bundle verified: {len(lines)} calibration images")
""",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a portable RKNN-Toolkit2 2.3.2 conversion bundle.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "yolov5n_adapted_v3" / "best.onnx",
    )
    parser.add_argument(
        "--anchors",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "yolov5n_adapted_v3" / "RK_anchors.txt",
    )
    parser.add_argument(
        "--quantization-list",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "yolov5n_adapted_v3" / "quantization.txt",
    )
    parser.add_argument("--toolkit-repo", type=Path, required=True)
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        help="Optional ONNX regression report to record in the portable bundle manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "rknn_conversion_bundle_v3",
    )
    args = parser.parse_args()

    required = (args.onnx, args.anchors, args.quantization_list)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError(f"missing model input: {[str(path) for path in required if not path.is_file()]}")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty bundle: {args.output}")
    evaluation = None
    if args.evaluation_report is not None:
        if not args.evaluation_report.is_file():
            raise FileNotFoundError(args.evaluation_report)
        evaluation_payload = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
        report_model = Path(str(evaluation_payload.get("model", "")))
        if not report_model.is_file() or report_model.resolve() != args.onnx.resolve():
            raise RuntimeError("evaluation report does not refer to the selected ONNX model")
        evaluation = {
            "report": str(args.evaluation_report.resolve()),
            "report_sha256": sha256(args.evaluation_report),
            "purpose": evaluation_payload.get("purpose"),
            "selected_threshold": evaluation_payload.get("selected_threshold"),
            "independent_acceptance_passed": evaluation_payload.get(
                "independent_acceptance_passed", False
            ),
        }
    commit = subprocess.check_output(
        ["git", "-C", str(args.toolkit_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_TOOLKIT_COMMIT:
        raise RuntimeError(f"expected RKNN-Toolkit2 v2.3.2 commit {EXPECTED_TOOLKIT_COMMIT}, got {commit}")

    toolkit_wheel = args.toolkit_repo / TOOLKIT_WHEEL
    toolkit_requirements = args.toolkit_repo / TOOLKIT_REQUIREMENTS
    arm_toolkit_wheel = args.toolkit_repo / ARM_TOOLKIT_WHEEL
    arm_toolkit_requirements = args.toolkit_repo / ARM_TOOLKIT_REQUIREMENTS
    lite_wheel = args.toolkit_repo / LITE_WHEEL
    official_license = args.toolkit_repo / "LICENSE"
    if any(
        not path.is_file()
        for path in (
            toolkit_wheel,
            toolkit_requirements,
            arm_toolkit_wheel,
            arm_toolkit_requirements,
            lite_wheel,
            official_license,
        )
    ):
        raise FileNotFoundError("official RKNN-Toolkit2 2.3.2 package files are incomplete")

    image_paths = [Path(line.strip()) for line in args.quantization_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(image_paths) != 300 or len({path.resolve() for path in image_paths}) != 300:
        raise RuntimeError("quantization list must contain 300 unique images")
    for path in image_paths:
        if not path.is_file() or path.parent.name != "train" or path.parent.parent.name != "images":
            raise RuntimeError(f"calibration image is missing or not from images/train: {path}")

    for relative in (
        "model", "calibration_images", "toolkit", "toolkit_arm64", "board_wheel", "output"
    ):
        (args.output / relative).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.onnx, args.output / "model" / "best.onnx")
    shutil.copy2(args.anchors, args.output / "model" / "RK_anchors.txt")
    shutil.copy2(PROJECT_ROOT / "scripts" / "convert_rknn.py", args.output / "convert_rknn.py")
    shutil.copy2(toolkit_wheel, args.output / "toolkit" / toolkit_wheel.name)
    shutil.copy2(toolkit_requirements, args.output / "toolkit" / toolkit_requirements.name)
    shutil.copy2(arm_toolkit_wheel, args.output / "toolkit_arm64" / arm_toolkit_wheel.name)
    shutil.copy2(
        arm_toolkit_requirements,
        args.output / "toolkit_arm64" / arm_toolkit_requirements.name,
    )
    shutil.copy2(lite_wheel, args.output / "board_wheel" / lite_wheel.name)
    shutil.copy2(official_license, args.output / "toolkit" / "LICENSE")

    portable_lines: list[str] = []
    image_digest = hashlib.sha256()
    for index, source in enumerate(image_paths):
        suffix = source.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp"}:
            raise RuntimeError(f"unsupported calibration image extension: {source}")
        destination = args.output / "calibration_images" / f"{index:03d}{suffix}"
        shutil.copy2(source, destination)
        relative = destination.relative_to(args.output).as_posix()
        portable_lines.append(relative)
        image_digest.update(bytes.fromhex(sha256(destination)))
    (args.output / "dataset.txt").write_text("\n".join(portable_lines) + "\n", encoding="utf-8", newline="\n")
    write_scripts(args.output)

    portable_files = (
        "model/best.onnx",
        "model/RK_anchors.txt",
        "convert_rknn.py",
        f"toolkit/{toolkit_wheel.name}",
        f"toolkit/{toolkit_requirements.name}",
        f"toolkit_arm64/{arm_toolkit_wheel.name}",
        f"toolkit_arm64/{arm_toolkit_requirements.name}",
        "toolkit/LICENSE",
        f"board_wheel/{lite_wheel.name}",
        "dataset.txt",
        "run_conversion.sh",
        "run_conversion_arm64.sh",
        "run_conversion_existing_env.sh",
        "install_board_lite.sh",
        "verify_bundle.py",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "RK3588 FP16 and INT8 conversion for the v4 incremental model",
        "acceptance_status": "pending_independent_board_validation",
        "target_platform": "rk3588",
        "toolkit": {
            "version": "2.3.2",
            "official_repository": "https://github.com/airockchip/rknn-toolkit2",
            "commit": commit,
            "conversion_python": "3.10 on x86_64 Linux or aarch64 Linux",
            "runtime_python": "3.10 aarch64 Linux",
        },
        "model": {
            "source": str(args.onnx.resolve()),
            "sha256": sha256(args.onnx),
            "input": [1, 3, 640, 640],
            "outputs": [[1, 18, 80, 80], [1, 18, 40, 40], [1, 18, 20, 20]],
            "postprocess_in_model": False,
        },
        "calibration": {
            "images": len(image_paths),
            "source_split": "train",
            "uses_validation_or_test": False,
            "portable_image_hash_aggregate": image_digest.hexdigest(),
            "dataset_txt_sha256": sha256(args.output / "dataset.txt"),
        },
        "onnx_regression": evaluation,
        "portable_file_sha256": {
            relative: sha256(args.output / relative) for relative in portable_files
        },
        "expected_outputs": [
            "output/steel_ball_yolov5n_fp16.rknn",
            "output/steel_ball_yolov5n_int8.rknn",
        ],
    }
    (args.output / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
