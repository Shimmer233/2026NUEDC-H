from __future__ import annotations

from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from scripts.export_rknn_onnx import EXPECTED_ANCHORS, verify_onnx_contract


def write_model(path: Path, output_channels: int = 18) -> None:
    graph = helper.make_graph(
        [],
        "contract",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [
            helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, output_channels, 80, 80]),
            helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1, 18, 40, 40]),
            helper.make_tensor_value_info("output2", TensorProto.FLOAT, [1, 18, 20, 20]),
        ],
    )
    onnx.save(helper.make_model(graph), str(path))


def test_verify_onnx_contract_accepts_deployment_shapes(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    anchors = tmp_path / "RK_anchors.txt"
    write_model(model)
    anchors.write_text("\n".join(str(value) for value in EXPECTED_ANCHORS), encoding="ascii")
    result = verify_onnx_contract(model, anchors)
    assert result["output_shapes"][0] == [1, 18, 80, 80]


def test_verify_onnx_contract_rejects_changed_head(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    anchors = tmp_path / "RK_anchors.txt"
    write_model(model, output_channels=21)
    anchors.write_text("\n".join(str(value) for value in EXPECTED_ANCHORS), encoding="ascii")
    with pytest.raises(RuntimeError, match="output shapes"):
        verify_onnx_contract(model, anchors)
