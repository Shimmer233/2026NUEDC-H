from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from steel_ball.rknn_postprocess import decode_yolov5_heads  # noqa: E402
from steel_ball.dataset import verify_split_review  # noqa: E402
from steel_ball.vision import letterbox_black, rknn_nhwc_batch, unletterbox_box  # noqa: E402


class OnnxBackend:
    def __init__(self, model: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("install onnxruntime or onnxruntime-gpu") from error
        self.session = ort.InferenceSession(str(model), providers=ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, rgb: np.ndarray) -> list[np.ndarray]:
        tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return self.session.run(None, {self.input_name: tensor})

    def close(self) -> None:
        pass


class RknnLiteBackend:
    def __init__(self, model: Path) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError as error:
            raise RuntimeError("install the RK3588 rknn_toolkit_lite2 2.3.2 wheel") from error
        self.runtime = RKNNLite(verbose=False)
        if self.runtime.load_rknn(str(model)) != 0:
            raise RuntimeError(f"unable to load {model}")
        core_mask = getattr(RKNNLite, "NPU_CORE_0_1_2", RKNNLite.NPU_CORE_AUTO)
        if self.runtime.init_runtime(core_mask=core_mask) != 0:
            raise RuntimeError("RKNN runtime initialization failed")

    def infer(self, rgb: np.ndarray) -> list[np.ndarray]:
        outputs = self.runtime.inference(
            inputs=[rknn_nhwc_batch(rgb)], data_format=["nhwc"]
        )
        if outputs is None:
            raise RuntimeError("RKNN inference returned no outputs")
        return outputs

    def close(self) -> None:
        self.runtime.release()


def write_prediction(path: Path, detections, width: int, height: int, meta) -> None:
    lines: list[str] = []
    for detection in detections:
        x1, y1, x2, y2 = unletterbox_box(detection.box, meta)
        center_x = (x1 + x2) / 2 / width
        center_y = (y1 + y2) / 2 / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(
            f"{detection.class_id} {center_x:.8f} {center_y:.8f} "
            f"{box_width:.8f} {box_height:.8f} {detection.confidence:.8f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ONNX or RKNN on a complete dataset split.")
    parser.add_argument("--backend", choices=("onnx", "rknn-lite"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\University\NUEDC\datasets\steel_ball_v2"),
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    images = sorted((args.dataset / "images" / args.split).glob("*.jpg"))
    if not images:
        raise RuntimeError(f"dataset split is empty: {args.split}")
    reviewed, reason = verify_split_review(args.dataset, args.split)
    if not reviewed:
        raise RuntimeError(f"inference blocked: {reason}")
    if args.output.exists():
        resolved = args.output.resolve()
        if PROJECT_ROOT.resolve() not in resolved.parents:
            raise RuntimeError(f"refusing to replace output outside project: {resolved}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    backend = OnnxBackend(args.model) if args.backend == "onnx" else RknnLiteBackend(args.model)
    latencies: list[float] = []
    try:
        for index, image_path in enumerate(images, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"unable to read {image_path}")
            model_input, meta = letterbox_black(image, 640)
            rgb = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            outputs = backend.infer(rgb)
            latencies.append((time.perf_counter() - started) * 1000.0)
            detections = decode_yolov5_heads(
                outputs,
                input_size=640,
                confidence_threshold=args.confidence,
                iou_threshold=args.iou,
            )
            write_prediction(
                args.output / f"{image_path.stem}.txt",
                detections,
                image.shape[1],
                image.shape[0],
                meta,
            )
            if index % 500 == 0:
                print(f"inferred {index}/{len(images)}")
    finally:
        backend.close()
    timings = np.asarray(latencies)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "images": len(images),
        "latency_ms_median": float(np.median(timings)),
        "latency_ms_p95": float(np.percentile(timings, 95)),
        "inference_fps_from_median": float(1000.0 / np.median(timings)),
    }
    (args.output / "inference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
