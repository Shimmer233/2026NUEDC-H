from __future__ import annotations

import sys
from pathlib import Path

from scripts import train_model


def test_incremental_arguments_do_not_enable_resume(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_model.py",
            "--init-weights",
            str(checkpoint),
            "--epochs",
            "80",
            "--patience",
            "15",
            "--batch-size",
            "32",
            "--seed",
            "42",
            "--noautoanchor",
        ],
    )
    args = train_model.parse_args()
    assert args.init_weights == checkpoint
    assert args.epochs == 80
    assert args.patience == 15
    assert args.batch_size == 32
    assert args.seed == 42
    assert args.noautoanchor
    assert not hasattr(args, "resume")
