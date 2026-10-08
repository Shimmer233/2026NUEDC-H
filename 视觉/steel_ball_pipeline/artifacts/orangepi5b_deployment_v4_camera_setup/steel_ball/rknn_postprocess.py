from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


DEFAULT_ANCHORS = np.asarray(
    [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Detection:
    box: np.ndarray
    confidence: float
    class_id: int = 0


def _head_as_anchor_grid(output: np.ndarray, class_count: int) -> np.ndarray:
    values_per_anchor = class_count + 5
    array = np.asarray(output)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[:2] == (3, values_per_anchor):
        return array.astype(np.float32, copy=False)
    if array.ndim != 3:
        raise ValueError(f"unsupported RKNN output shape: {array.shape}")
    expected_channels = 3 * values_per_anchor
    if array.shape[0] == expected_channels:
        chw = array
    elif array.shape[-1] == expected_channels:
        chw = array.transpose(2, 0, 1)
    else:
        raise ValueError(
            f"expected {expected_channels} channels for {class_count} classes, got {array.shape}"
        )
    return chw.reshape(3, values_per_anchor, chw.shape[1], chw.shape[2]).astype(
        np.float32, copy=False
    )


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        ix1 = np.maximum(x1[current], x1[remaining])
        iy1 = np.maximum(y1[current], y1[remaining])
        ix2 = np.minimum(x2[current], x2[remaining])
        iy2 = np.minimum(y2[current], y2[remaining])
        intersection = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_yolov5_heads(
    outputs: Sequence[np.ndarray],
    input_size: int = 640,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_count: int = 1,
    anchors: np.ndarray = DEFAULT_ANCHORS,
) -> list[Detection]:
    """Decode sigmoid RKNN YOLOv5 heads produced by the Rockchip export branch."""
    if len(outputs) != 3:
        raise ValueError(f"expected three detection heads, got {len(outputs)}")
    heads = [_head_as_anchor_grid(output, class_count) for output in outputs]
    heads.sort(key=lambda head: head.shape[-1], reverse=True)
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []
    for head_index, head in enumerate(heads):
        _, _, grid_height, grid_width = head.shape
        prediction = head.transpose(0, 2, 3, 1)
        grid_y, grid_x = np.meshgrid(
            np.arange(grid_height, dtype=np.float32),
            np.arange(grid_width, dtype=np.float32),
            indexing="ij",
        )
        grid = np.stack((grid_x, grid_y), axis=-1)[None, ...]
        stride = np.asarray(
            [input_size / grid_width, input_size / grid_height], dtype=np.float32
        )
        center = (prediction[..., 0:2] * 2.0 - 0.5 + grid) * stride
        size = (prediction[..., 2:4] * 2.0) ** 2 * anchors[head_index, :, None, None, :]
        boxes = np.concatenate((center - size / 2.0, center + size / 2.0), axis=-1)
        class_scores = prediction[..., 5:] * prediction[..., 4:5]
        class_ids = class_scores.argmax(axis=-1)
        scores = class_scores.max(axis=-1)
        selected = scores >= confidence_threshold
        if np.any(selected):
            all_boxes.append(boxes[selected])
            all_scores.append(scores[selected])
            all_classes.append(class_ids[selected])
    if not all_boxes:
        return []
    boxes = np.clip(np.concatenate(all_boxes), 0, input_size - 1)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)
    detections: list[Detection] = []
    for class_id in np.unique(classes):
        indices = np.flatnonzero(classes == class_id)
        for kept in _nms(boxes[indices], scores[indices], iou_threshold):
            source_index = indices[kept]
            detections.append(
                Detection(boxes[source_index], float(scores[source_index]), int(class_id))
            )
    return sorted(detections, key=lambda detection: detection.confidence, reverse=True)
