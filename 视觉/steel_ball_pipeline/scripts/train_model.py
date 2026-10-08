from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.dataset import verify_approval  # noqa: E402
from steel_ball.training_view import build_training_view  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an isolated, review-approved steel-ball model.")
    parser.add_argument("--model", choices=("yolov5n", "yolov5s"), default="yolov5n")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\University\NUEDC\datasets\steel_ball_v2"),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--init-weights",
        type=Path,
        help="Start a new optimizer run from this checkpoint; this never enables --resume.",
    )
    parser.add_argument(
        "--noautoanchor",
        action="store_true",
        help="Keep the deployment decoder's existing YOLOv5 anchors.",
    )
    parser.add_argument("--run-name")
    parser.add_argument(
        "--yolov5n-test-report",
        type=Path,
        help="Required for yolov5s; must show YOLOv5n test recall below 99 percent.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model == "yolov5s":
        if args.yolov5n_test_report is None or not args.yolov5n_test_report.is_file():
            raise RuntimeError("YOLOv5s is blocked until --yolov5n-test-report is provided")
        nano_report = json.loads(args.yolov5n_test_report.read_text(encoding="utf-8"))
        if "subsets" in nano_report:
            nano_recall = float(nano_report["subsets"]["all_ball_frames"]["recall"])
        elif "thresholds" in nano_report:
            nano_recall = max(
                float(result["combined"]["ball_recall"])
                for result in nano_report["thresholds"].values()
            )
        else:
            raise RuntimeError("unrecognized YOLOv5n test report schema")
        if nano_recall >= 0.99:
            raise RuntimeError(f"YOLOv5n recall is {nano_recall:.4f}; YOLOv5s is not required")
    approved, reason = verify_approval(args.dataset)
    if not approved:
        raise RuntimeError(f"training blocked: {reason}")
    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    yolov5 = PROJECT_ROOT / "vendor" / "yolov5-rkopt"
    dirty = git_output(yolov5, "status", "--porcelain")
    if dirty:
        raise RuntimeError("Rockchip YOLOv5 vendor checkout is not clean")
    if args.init_weights is not None:
        weights = args.init_weights.resolve()
        if not weights.is_file():
            raise FileNotFoundError(weights)
        initial_weight = {
            "kind": "fine_tune_checkpoint",
            "path": str(weights),
            "sha256": file_sha256(weights),
            "optimizer_resumed": False,
        }
    else:
        weights = PROJECT_ROOT / "weights" / f"{args.model}.pt"
        weight_manifest = PROJECT_ROOT / "weights" / f"{args.model}.json"
        if not weights.exists() or not weight_manifest.exists():
            raise RuntimeError(
                f"official weights are missing; run: python scripts/download_official_weights.py {args.model}"
            )
        official = json.loads(weight_manifest.read_text(encoding="utf-8"))
        if official.get("sha256") != file_sha256(weights):
            raise RuntimeError("official weight checksum does not match its download manifest")
        if "github.com/ultralytics/yolov5/releases/download/v7.0/" not in official.get("source", ""):
            raise RuntimeError("weight source is not the official Ultralytics YOLOv5 v7.0 release")
        initial_weight = {
            "kind": "official_yolov5_v7",
            "path": str(weights.resolve()),
            "sha256": file_sha256(weights),
            "manifest": official,
            "optimizer_resumed": False,
        }

    batch_size = args.batch_size or (32 if args.model == "yolov5n" else 16)
    training_data_yaml = build_training_view(args.dataset, seed=args.seed, fraction=0.70)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"steel_ball_{args.model}_seed{args.seed}_{timestamp}"
    run_root = PROJECT_ROOT / "runs" / "train"
    command = [
        sys.executable,
        str(yolov5 / "train.py"),
        "--weights",
        str(weights),
        "--cfg",
        str(yolov5 / "models" / f"{args.model}.yaml"),
        "--data",
        str(training_data_yaml),
        "--hyp",
        str(PROJECT_ROOT / "config" / "hyp.steel_ball.yaml"),
        "--imgsz",
        "640",
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
        "--device",
        args.device,
        "--project",
        str(run_root),
        "--name",
        run_name,
    ]
    if args.noautoanchor:
        command.append("--noautoanchor")
    manifest_dir = PROJECT_ROOT / "runs" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "model": args.model,
        "dataset": str(args.dataset),
        "training_data_yaml": str(training_data_yaml),
        "training_view_manifest": json.loads(
            (args.dataset / "generated" / "training_view" / "manifest.json").read_text(
                encoding="utf-8"
            )
        ),
        "dataset_approval": json.loads(
            (args.dataset / ".review-approved.json").read_text(encoding="utf-8")
        ),
        "initial_weight": initial_weight,
        "optimizer_resumed": False,
        "anchors_frozen": args.noautoanchor,
        "vendor_commit": git_output(yolov5, "rev-parse", "HEAD"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_force_no_weights_only_load": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pip_freeze": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, encoding="utf-8"
        ).splitlines(),
    }
    source_inventory = args.dataset / "manifests" / "source_inventory.json"
    if source_inventory.exists():
        manifest["source_inventory"] = json.loads(source_inventory.read_text(encoding="utf-8"))
    manifest_path = manifest_dir / f"{run_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Launching isolated training command:")
    print(subprocess.list2cmdline(command))
    environment = os.environ.copy()
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    subprocess.run(command, cwd=yolov5, env=environment, check=True)
    print(f"Training complete: {run_root / run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
