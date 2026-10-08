from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS = PROJECT_ROOT / "artifacts" / "manual_deployment_v4_incremental" / "scripts"
FIELDS = (
    "timestamp_ns",
    "frame_id",
    "detected",
    "confidence",
    "x_px",
    "y_px",
    "s_cm",
    "velocity_cm_s",
    "predicted",
)
TEST_FRAMES = 3000


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_stage_manifest(tmp_path: Path, suffix: str = "a") -> Path:
    path = tmp_path / "stage.json"
    path.write_text(
        json.dumps(
            {
                "active_model_changed": False,
                "nonce": suffix,
                "files": {
                    "models/steel_ball_yolov5n_v4_incremental_int8.rknn": {
                        "size": 4,
                        "sha256": f"int8-{suffix}",
                    },
                    "config/vision_v4_incremental_staged.yaml": {
                        "size": 4,
                        "sha256": f"config-{suffix}",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def write_run(
    runs_root: Path,
    stage_manifest: Path,
    scenario: str,
    candidates: list[bool],
    *,
    detected: list[bool] | None = None,
    predicted: list[bool] | None = None,
    positions: list[bool] | None = None,
    log_text: str = "runtime completed\n",
) -> Path:
    index = len(list(runs_root.glob(f"*_{scenario}")))
    run_dir = runs_root / f"{index:02d}_{scenario}"
    run_dir.mkdir(parents=True)
    detected = candidates if detected is None else detected
    predicted = [False] * len(candidates) if predicted is None else predicted
    positions = [d or p for d, p in zip(detected, predicted)] if positions is None else positions
    assert len(candidates) == len(detected) == len(predicted) == len(positions)

    csv_path = run_dir / "trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index, (candidate, measurement, prediction, position) in enumerate(
            zip(candidates, detected, predicted, positions)
        ):
            writer.writerow(
                {
                    "timestamp_ns": index * 12_500_000,
                    "frame_id": index,
                    "detected": int(measurement),
                    "confidence": "0.9" if candidate else "",
                    "x_px": "320" if candidate else "",
                    "y_px": "200" if candidate else "",
                    "s_cm": "12.5" if position else "",
                    "velocity_cm_s": "0.0" if position else "",
                    "predicted": int(prediction),
                }
            )
    csv_path.with_suffix(".performance.json").write_text(
        json.dumps(
            {
                "duration_seconds": 60.0,
                "processed_frames": len(candidates),
                "fusion_fps_median": 50.0,
            }
        ),
        encoding="utf-8",
    )
    log_path = run_dir / "run.log"
    log_path.write_text(log_text, encoding="utf-8")
    (run_dir / "attempt.json").write_text(
        json.dumps(
            {
                "scenario": scenario,
                "camera": "/dev/video0",
                "stage_manifest_sha256": file_hash(stage_manifest),
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "summarize_run.py"),
            "--csv",
            str(csv_path),
            "--log",
            str(log_path),
            "--stage-manifest",
            str(stage_manifest),
            "--camera",
            "/dev/video0",
            "--scenario",
            scenario,
            "--requested-duration",
            "60",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert result.returncode == (0 if summary["passed"] else 2), result.stderr
    return summary_path


def test_ball_summary_allows_at_most_three_consecutive_misses(tmp_path: Path) -> None:
    stage = write_stage_manifest(tmp_path)
    candidates = [True] * TEST_FRAMES
    candidates[400:403] = [False, False, False]

    summary = json.loads(
        write_run(tmp_path / "runs", stage, "fast", candidates).read_text(encoding="utf-8")
    )

    assert summary["passed"] is True
    assert summary["recall"] == 0.999
    assert summary["longest_consecutive_misses"] == 3


def test_ball_summary_rejects_four_consecutive_misses(tmp_path: Path) -> None:
    stage = write_stage_manifest(tmp_path)
    candidates = [True] * TEST_FRAMES
    candidates[400:404] = [False, False, False, False]

    summary = json.loads(
        write_run(tmp_path / "runs", stage, "normal", candidates).read_text(encoding="utf-8")
    )

    assert summary["passed"] is False
    assert summary["longest_consecutive_misses"] == 4


def test_empty_summary_enforces_raw_and_tracking_outputs(tmp_path: Path) -> None:
    stage = write_stage_manifest(tmp_path)
    passing = [False] * TEST_FRAMES
    passing[20:23] = [True, True, True]
    failing = [False] * TEST_FRAMES
    failing[20:24] = [True, True, True, True]

    pass_summary = json.loads(
        write_run(
            tmp_path / "pass_runs",
            stage,
            "empty",
            passing,
            detected=[False] * TEST_FRAMES,
            positions=[False] * TEST_FRAMES,
        ).read_text(encoding="utf-8")
    )
    fail_summary = json.loads(
        write_run(
            tmp_path / "fail_runs",
            stage,
            "empty",
            failing,
            detected=[False] * TEST_FRAMES,
            positions=[False] * TEST_FRAMES,
        ).read_text(encoding="utf-8")
    )
    tracked_summary = json.loads(
        write_run(
            tmp_path / "tracked_runs",
            stage,
            "empty",
            [False] * TEST_FRAMES,
            detected=[True] + [False] * (TEST_FRAMES - 1),
            positions=[True] + [False] * (TEST_FRAMES - 1),
        ).read_text(encoding="utf-8")
    )

    assert pass_summary["passed"] is True
    assert pass_summary["false_positives_per_1000_frames"] == 1.0
    assert fail_summary["passed"] is False
    assert tracked_summary["passed"] is False
    assert tracked_summary["tracked_detection_frames"] == 1


def test_runtime_warning_closes_run_gate(tmp_path: Path) -> None:
    stage = write_stage_manifest(tmp_path)
    summary = json.loads(
        write_run(
            tmp_path / "runs",
            stage,
            "normal",
            [True] * TEST_FRAMES,
            log_text="YOLO inference warning: simulated failure\n",
        ).read_text(encoding="utf-8")
    )

    assert summary["passed"] is False
    assert summary["log_gate_passed"] is False


def test_ball_confidence_without_position_output_is_rejected(tmp_path: Path) -> None:
    stage = write_stage_manifest(tmp_path)
    summary = json.loads(
        write_run(
            tmp_path / "runs",
            stage,
            "normal",
            [True] * TEST_FRAMES,
            detected=[False] * TEST_FRAMES,
            predicted=[False] * TEST_FRAMES,
            positions=[False] * TEST_FRAMES,
        ).read_text(encoding="utf-8")
    )

    assert summary["recall"] == 1.0
    assert summary["position_availability"] == 0.0
    assert summary["passed"] is False


def test_ball_prediction_cannot_hide_slow_measurement_updates(tmp_path: Path) -> None:
    stage = write_stage_manifest(tmp_path)
    measurements = [index % 4 == 0 for index in range(TEST_FRAMES)]
    summary = json.loads(
        write_run(
            tmp_path / "runs",
            stage,
            "fast",
            [True] * TEST_FRAMES,
            detected=measurements,
            predicted=[not value for value in measurements],
            positions=[True] * TEST_FRAMES,
        ).read_text(encoding="utf-8")
    )

    assert summary["position_availability"] == 1.0
    assert summary["accepted_measurement_hz"] == 12.5
    assert summary["passed"] is False


def test_low_fusion_fps_closes_run_gate(tmp_path: Path) -> None:
    stage = write_stage_manifest(tmp_path)
    summary_path = write_run(
        tmp_path / "runs", stage, "normal", [True] * TEST_FRAMES
    )
    performance_path = summary_path.parent / "trajectory.performance.json"
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    performance["fusion_fps_median"] = 1.0
    performance_path.write_text(json.dumps(performance), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "summarize_run.py"),
            "--csv",
            str(summary_path.parent / "trajectory.csv"),
            "--log",
            str(summary_path.parent / "run.log"),
            "--stage-manifest",
            str(stage),
            "--camera",
            "/dev/video0",
            "--scenario",
            "normal",
            "--requested-duration",
            "60",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert result.returncode == 2
    assert summary["passed"] is False
    assert summary["fusion_fps_gate_passed"] is False


def test_high_median_burst_does_not_hide_low_average_fps(tmp_path: Path) -> None:
    stage = write_stage_manifest(tmp_path)
    summary = json.loads(
        write_run(
            tmp_path / "runs", stage, "normal", [True] * 16
        ).read_text(encoding="utf-8")
    )

    assert summary["performance"]["fusion_fps_median"] == 50.0
    assert summary["average_fps"] < 1.0
    assert summary["fusion_fps_gate_passed"] is False
    assert summary["passed"] is False


def test_activation_gate_requires_latest_run_for_current_stage(tmp_path: Path) -> None:
    stage = write_stage_manifest(tmp_path)
    runs = tmp_path / "runs"
    write_run(runs, stage, "normal", [True] * TEST_FRAMES)
    write_run(runs, stage, "fast", [True] * TEST_FRAMES)
    write_run(
        runs,
        stage,
        "empty",
        [False] * TEST_FRAMES,
        detected=[False] * TEST_FRAMES,
        positions=[False] * TEST_FRAMES,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "check_activation_gate.py"),
            str(runs),
            "--stage-manifest",
            str(stage),
            "--camera",
            "/dev/video0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Activation gate passed" in result.stdout


def test_activation_gate_rejects_reports_from_previous_stage(tmp_path: Path) -> None:
    stage = write_stage_manifest(tmp_path, "a")
    runs = tmp_path / "runs"
    write_run(runs, stage, "normal", [True] * TEST_FRAMES)
    write_run(runs, stage, "fast", [True] * TEST_FRAMES)
    write_run(
        runs,
        stage,
        "empty",
        [False] * TEST_FRAMES,
        detected=[False] * TEST_FRAMES,
        positions=[False] * TEST_FRAMES,
    )
    write_stage_manifest(tmp_path, "b")

    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "check_activation_gate.py"),
            str(runs),
            "--stage-manifest",
            str(stage),
            "--camera",
            "/dev/video0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "another staging operation" in result.stderr


def test_activation_gate_rejects_incomplete_latest_attempt(tmp_path: Path) -> None:
    stage = write_stage_manifest(tmp_path)
    runs = tmp_path / "runs"
    write_run(runs, stage, "normal", [True] * TEST_FRAMES)
    write_run(runs, stage, "fast", [True] * TEST_FRAMES)
    write_run(
        runs,
        stage,
        "empty",
        [False] * TEST_FRAMES,
        detected=[False] * TEST_FRAMES,
        positions=[False] * TEST_FRAMES,
    )
    incomplete = runs / "99_normal"
    incomplete.mkdir()
    attempt = incomplete / "attempt.json"
    attempt.write_text(
        json.dumps(
            {
                "scenario": "normal",
                "camera": "/dev/video0",
                "stage_manifest_sha256": file_hash(stage),
            }
        ),
        encoding="utf-8",
    )
    future = time.time() + 5
    os.utime(incomplete, (future, future))

    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "check_activation_gate.py"),
            str(runs),
            "--stage-manifest",
            str(stage),
            "--camera",
            "/dev/video0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "incomplete" in result.stderr


def test_activation_gate_rejects_wrong_summary_scenario(tmp_path: Path) -> None:
    stage = write_stage_manifest(tmp_path)
    runs = tmp_path / "runs"
    write_run(runs, stage, "normal", [True] * TEST_FRAMES)
    fast_summary = write_run(runs, stage, "fast", [True] * TEST_FRAMES)
    write_run(
        runs,
        stage,
        "empty",
        [False] * TEST_FRAMES,
        detected=[False] * TEST_FRAMES,
        positions=[False] * TEST_FRAMES,
    )
    payload = json.loads(fast_summary.read_text(encoding="utf-8"))
    payload["scenario"] = "normal"
    fast_summary.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "check_activation_gate.py"),
            str(runs),
            "--stage-manifest",
            str(stage),
            "--camera",
            "/dev/video0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "wrong scenario identity" in result.stderr


def test_verify_stage_detects_tampering(tmp_path: Path) -> None:
    app = tmp_path / "app"
    model = app / "models" / "steel_ball_yolov5n_v4_incremental_int8.rknn"
    config = app / "config" / "vision_v4_incremental_staged.yaml"
    active_config = app / "config" / "vision.yaml"
    calibration = app / "config" / "track_calibration.yaml"
    geometry = app / "config" / "track_geometry.yaml"
    model.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    config.write_text("model: staged\n", encoding="utf-8")
    active_config.write_text("model: active\n", encoding="utf-8")
    calibration.write_text("points: []\n", encoding="utf-8")
    geometry.write_text("corners: []\n", encoding="utf-8")
    manifest = app / "models" / "steel_ball_v4_incremental.stage.json"
    manifest.write_text(
        json.dumps(
            {
                "active_model_changed": False,
                "source_config_sha256": file_hash(active_config),
                "runtime_dependencies": {
                    calibration.relative_to(app).as_posix(): {
                        "size": calibration.stat().st_size,
                        "sha256": file_hash(calibration),
                    },
                    geometry.relative_to(app).as_posix(): {
                        "size": geometry.stat().st_size,
                        "sha256": file_hash(geometry),
                    },
                },
                "files": {
                    model.relative_to(app).as_posix(): {
                        "size": model.stat().st_size,
                        "sha256": file_hash(model),
                    },
                    config.relative_to(app).as_posix(): {
                        "size": config.stat().st_size,
                        "sha256": file_hash(config),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    command = [sys.executable, str(TOOLS / "verify_stage.py"), str(app), str(manifest)]
    assert subprocess.run(command, check=False).returncode == 0
    active_config.write_text("model: changed\n", encoding="utf-8")
    assert subprocess.run(command, check=False).returncode != 0
    active_config.write_text("model: active\n", encoding="utf-8")
    model.write_bytes(b"tampered")
    assert subprocess.run(command, check=False).returncode != 0


def test_complete_backup_manifest_verifies_restored_config_and_models(
    tmp_path: Path,
) -> None:
    backup = tmp_path / "backup"
    (backup / "models").mkdir(parents=True)
    (backup / "config").mkdir()
    (backup / "models" / "model.rknn").write_bytes(b"rknn")
    (backup / "models" / "model.json").write_text("{}\n", encoding="utf-8")
    (backup / "config" / "vision.yaml").write_text("model: old\n", encoding="utf-8")
    make_command = [
        sys.executable,
        str(TOOLS / "make_backup_manifest.py"),
        str(backup),
        "--purpose",
        "test backup",
    ]
    assert subprocess.run(make_command, check=False).returncode == 0
    verify_command = [sys.executable, str(TOOLS / "verify_backup.py"), str(backup)]
    assert subprocess.run(verify_command, check=False).returncode == 0

    restored = tmp_path / "restored"
    shutil.copytree(backup / "models", restored / "models")
    shutil.copytree(backup / "config", restored / "config")
    restored_command = [*verify_command, "--restored-root", str(restored)]
    assert subprocess.run(restored_command, check=False).returncode == 0
    (restored / "config" / "vision.yaml").write_text(
        "model: corrupted\n", encoding="utf-8"
    )
    assert subprocess.run(restored_command, check=False).returncode != 0
