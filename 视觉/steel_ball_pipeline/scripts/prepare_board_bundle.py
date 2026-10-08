from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Orange Pi 5B hybrid vision/control deployment bundle."
    )
    parser.add_argument(
        "--conversion-bundle",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "rknn_conversion_bundle_v3_fixed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "orangepi5b_deployment_v5_hybrid_control"
        ),
    )
    parser.add_argument(
        "--refresh-generated-v5",
        action="store_true",
        help="Refresh only an existing bundle whose manifest declares version 5.0.0.",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        manifest_path = args.output / "deployment_manifest.json"
        if not args.refresh_generated_v5 or not manifest_path.is_file():
            raise RuntimeError(f"refusing to overwrite non-empty bundle: {args.output}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("bundle_version") != "5.0.0":
            raise RuntimeError("refresh is allowed only for a generated v5 bundle")

    lite_wheels = list((args.conversion_bundle / "board_wheel").glob("*.whl"))
    if len(lite_wheels) != 1:
        raise RuntimeError("conversion bundle must contain exactly one RKNNLite2 wheel")
    conversion_output = args.conversion_bundle / "output"
    int8_model = conversion_output / "steel_ball_yolov5n_int8.rknn"
    if not int8_model.is_file():
        raise RuntimeError(f"INT8 model is missing: {int8_model}")

    for relative in (
        "board",
        "steel_ball",
        "config",
        "scripts",
        "board_wheel",
        "models",
        "runs",
    ):
        (args.output / relative).mkdir(parents=True, exist_ok=True)

    for source in (PROJECT_ROOT / "steel_ball").glob("*.py"):
        shutil.copy2(source, args.output / "steel_ball" / source.name)
    shutil.copy2(
        PROJECT_ROOT / "board" / "steel_ball_tracker.py", args.output / "board"
    )
    shutil.copy2(
        PROJECT_ROOT / "board" / "requirements-board.txt", args.output / "board"
    )
    for script_name in (
        "calibrate_track.py",
        "calibrate_track_geometry.py",
        "calibrate_actuator.py",
        "manual_motor_limits.py",
        "apply_motor_safety_update.py",
        "fit_control_gains.py",
        "setup_camera.py",
    ):
        shutil.copy2(PROJECT_ROOT / "scripts" / script_name, args.output / "scripts")
    shutil.copy2(lite_wheels[0], args.output / "board_wheel" / lite_wheels[0].name)
    shutil.copy2(
        PROJECT_ROOT / "config" / "actuator_calibration.example.yaml",
        args.output / "config",
    )

    config = yaml.safe_load(
        (PROJECT_ROOT / "config" / "board.yaml").read_text(encoding="utf-8")
    )
    config["model"] = "models/steel_ball_yolov5n_int8.rknn"
    config["calibration"] = "config/track_calibration.yaml"
    config["geometry"] = "config/track_geometry.yaml"
    config["motor"]["calibration"] = "config/actuator_calibration.yaml"
    (args.output / "config" / "board.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n"
    )

    copied_models: list[str] = []
    for source in conversion_output.iterdir():
        if source.is_file() and (
            source.suffix in {".rknn", ".json"} or source.name == "SHA256SUMS.txt"
        ):
            shutil.copy2(source, args.output / "models" / source.name)
            if source.suffix == ".rknn":
                copied_models.append(source.name)

    write_executable(
        args.output / "install.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sudo apt-get update
sudo apt-get install -y python3-venv python3-opencv python3-yaml python3-psutil python3-serial
if ! command -v v4l2-ctl >/dev/null 2>&1; then
  sudo apt-get install -y --no-upgrade v4l-utils
fi
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps board_wheel/{lite_wheels[0].name}
.venv/bin/python - <<'PY'
import cv2, numpy, serial, yaml
from rknnlite.api import RKNNLite
print("OpenCV", cv2.__version__)
print("NumPy", numpy.__version__)
print("pyserial", serial.__version__)
print("RKNNLite2 import OK", RKNNLite)
PY
""",
    )
    write_executable(
        args.output / "list_devices.sh",
        """#!/usr/bin/env bash
set -euo pipefail
v4l2-ctl --list-devices
echo
echo "Stable USB camera paths:"
find /dev/v4l/by-id -maxdepth 1 -type l -print 2>/dev/null || true
echo
ls -l /dev/ttyS0 /dev/ttyS1 2>/dev/null || true
echo
for device in /dev/video*; do
  [[ -e "$device" ]] || continue
  echo "===== $device ====="
  v4l2-ctl --device="$device" --list-formats-ext 2>/dev/null || true
done
""",
    )
    write_executable(
        args.output / "run_observe.sh",
        """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CAMERA="${1:-/dev/video0}"
exec .venv/bin/python board/steel_ball_tracker.py --config config/board.yaml --camera-device "$CAMERA" --control-mode observe
""",
    )
    write_executable(
        args.output / "run_detection_only.sh",
        """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CAMERA="${1:-/dev/video0}"
exec .venv/bin/python board/steel_ball_tracker.py --config config/board.yaml --camera-device "$CAMERA" --control-mode observe --detection-only
""",
    )
    write_executable(
        args.output / "run_dry_run.sh",
        """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CAMERA="${1:-/dev/video0}"
TARGET="${2:-12.5}"
exec .venv/bin/python board/steel_ball_tracker.py --config config/board.yaml --camera-device "$CAMERA" --control-mode dry-run --target-cm "$TARGET"
""",
    )
    write_executable(
        args.output / "run_identification.sh",
        """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CAMERA="${1:-/dev/video0}"
ANGLE="${2:-0.15}"
ARM="${3:-}"
if [[ "$ARM" != "--arm" ]]; then
  echo "Motor movement blocked. Pass --arm as the third argument after completing all safety gates." >&2
  exit 2
fi
exec .venv/bin/python board/steel_ball_tracker.py --config config/board.yaml --camera-device "$CAMERA" --control-mode identify --identify-angle-deg "$ANGLE" --arm
""",
    )
    write_executable(
        args.output / "run_balance.sh",
        """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CAMERA="${1:-/dev/video0}"
TARGET="${2:-12.5}"
ARM="${3:-}"
if [[ "$ARM" != "--arm" ]]; then
  echo "Balance blocked. Pass --arm as the third argument after zero and gain verification." >&2
  exit 2
fi
exec .venv/bin/python board/steel_ball_tracker.py --config config/board.yaml --camera-device "$CAMERA" --control-mode balance --target-cm "$TARGET" --arm
""",
    )

    readme = """# Orange Pi 5B 钢球 YOLO 检测与一维坐标控制 v5

本包默认只运行 `observe`，不会打开 `/dev/ttyS0`，也不会使能电机。钢球候选只接受 YOLO 检测，CV 仅负责轨道四点/厘米标定、坐标系和 YOLO 框内圆心精修，不再提供独立候选参与融合。视觉不测量水管水平。`/dev/ttyS1` 仅预留，当前程序不会打开它。

## 安全边界

- 保留并随时能触及电机物理断电开关。
- 软件急停会发送 `C5 01 FC C2 5C` 并锁定程序；随后必须立即物理断电，避免刹车状态长期发热。
- 正常启动永远不会发送 `F8`。只有人工执行一次性零点脚本时才允许清零。
- 零点断电保持验证失败时，只能使用人工调平后的当前启动会话零点；启动标记或实时计数不匹配时，`balance --arm` 和辨识运动都被代码禁止。
- `0xFB` 不进入自动流程。故障清除只能在断电检查、人工确认并重启后进行。

## 安装与纯视觉

```bash
chmod +x *.sh
./install.sh
./list_devices.sh
./run_detection_only.sh /dev/video0
./run_observe.sh /dev/video0
```

相机固定请求 640x480、MJPG、120 FPS。卡尔曼/控制循环 80 Hz，INT8 YOLO 固定目标 50 Hz，显示 30 Hz。异步调度按推理开始时间限频，不会把推理耗时重复加到周期中。先运行相机曝光工具，再依次标定轨道四点和厘米坐标：

```bash
.venv/bin/python scripts/setup_camera.py --device /dev/video0
.venv/bin/python scripts/calibrate_track_geometry.py --device /dev/video0
.venv/bin/python scripts/calibrate_track.py --device /dev/video0 --width 640 --height 480 --fps 120
```

厘米标定内部仍使用左端 `0 cm`、右端 `25 cm`。显示窗口自动转换为题图坐标 `x=s-12.5 cm`：水管中心为 `0`，左侧为负、右侧为正，并绘制 0.5 cm 小刻度、2 cm 数字标签和钢球实时一维坐标。控制器和 CSV 继续保留 `s=0..25 cm`，避免破坏既有数据格式。

## 电机零点与映射

1. 用手机水平仪将水管调平，确认物理急停可用。
2. 根据机构保守设置软限位后，执行下例。它依次发送位置模式、`F8` 和参数保存：

```bash
.venv/bin/python scripts/calibrate_actuator.py zero --confirm-level --min-count -500 --max-count 500
```

3. 保持机构不动并完整断电重启，然后执行：

```bash
.venv/bin/python scripts/calibrate_actuator.py verify --confirm-power-cycled --confirm-not-moved
```

如果驱动器明确不保存 `F8` 零点，不要修改 YAML 绕过验证。每次驱动器或 Orange Pi 上电后，用手机或机械基准调平并执行：

```bash
.venv/bin/python scripts/calibrate_actuator.py session-zero --confirm-level --confirm-supported
```

会话零点标记同时绑定当前 Linux 启动 ID 和实时电机计数。任一设备掉电或计数跳出软限位后，运动命令会重新锁定。IMU 不参与水管找平；它只预留给车体加速度前馈。

4. 只用 10 RPM、加速度 10 做正负小角度点动，手机记录角度并读取计数。异常时脚本发送 FC，随后必须物理断电。收集至少三个点后拟合：

```bash
.venv/bin/python scripts/calibrate_actuator.py jog --target-count 100 --arm
.venv/bin/python scripts/calibrate_actuator.py status
.venv/bin/python scripts/calibrate_actuator.py fit --point=-0.3:-300 --point=0:0 --point=0.3:300 --min-count -500 --max-count 500
```

示例数字不能直接用于你的机构。拟合后再次检查 `config/actuator_calibration.yaml` 的方向和软限位。

## 分阶段控制

先进行不发串口命令的 dry-run：

```bash
./run_dry_run.sh /dev/video0 12.5
```

确认轨迹、方向和软限位后，才允许小倾角辨识。第三个参数必须明确写 `--arm`：

```bash
./run_identification.sh /dev/video0 0.15 --arm
.venv/bin/python scripts/fit_control_gains.py
# 检查 runs/board/control_identification.json 后再写入：
.venv/bin/python scripts/fit_control_gains.py --confirm-write
```

闭环同样必须显式解锁：

```bash
./run_balance.sh /dev/video0 12.5 --arm
```

触摸轨道设置 1-24 cm 目标；红色 STOP 或键盘 `E` 触发锁定急停，`Q`/Esc 正常退出并尝试回水平零位后失能。`trajectory.csv` 保持九字段，`control_telemetry.csv` 记录控制器、电机和状态机数据。

## 硬件验收顺序

纯视觉 -> 永久零点或当前启动会话零点 -> 无球软限位 -> 带球 dry-run -> 0.15 度辨识 -> 低增益闭环。桌面单元测试不能替代这些硬件门槛。
"""
    (args.output / "README_CN.md").write_text(
        readme, encoding="utf-8", newline="\n"
    )
    update_readme = """# YOLO-only 视觉更新

此更新不更换 RKNN 模型、不重建 `.venv`、不执行电机命令。它完成以下调整：

- YOLO 成为唯一钢球检测来源，CV 候选不再参与融合。
- CV 仅用于轨道四点/厘米标定、坐标系和 YOLO 框内圆心精修。
- INT8 YOLO 固定目标 50 Hz，并修复推理耗时被重复加入调度周期的问题。
- 显示以水管中心为 `0` 的一维坐标轴，左负右正，显示实时 `x` 坐标。

把更新压缩包解压并覆盖到现有 `orangepi5b_deployment_v5_hybrid_control` 后，直接运行：

```bash
cd ~/orangepi5b_deployment_v5_hybrid_control
./run_observe.sh /dev/video0
```

若尚未生成 `config/track_calibration.yaml`，坐标值和坐标轴不会显示。先按主说明完成轨道四点和厘米标定。
"""
    (args.output / "UPDATE_YOLO_ONLY_CN.md").write_text(
        update_readme, encoding="utf-8", newline="\n"
    )
    session_zero_readme = """# 当前启动会话零点更新

本更新适用于 `F8` 清零后计数为 0、但驱动器掉电后零点恢复为其他数值的设备。它不会使用视觉或 IMU 推算水管水平。

每次驱动器或 Orange Pi 上电后：

1. 取出钢球并可靠支撑水管。
2. 将水管用手机水平仪或机械基准调平。
3. 执行：

```bash
.venv/bin/python scripts/calibrate_actuator.py session-zero --confirm-level --confirm-supported
```

成功必须显示 `SESSION ZERO READY: count=0`。随后可继续低速点动和角度映射。

安全门槛：

- 标记绑定当前 Linux boot ID，Orange Pi 重启后自动失效。
- 每次运动前读取电机实际计数；标记缺失、计数偏离零点或超出软限位时拒绝运动。
- 电机运行中若驱动器掉电并恢复到旧坐标，软限位检查触发锁定急停。
- 不允许手工把 `persistence_verified` 改为 `true`。
- IMU 只用于后续车体加速度前馈，不用于水管找平。
"""
    (args.output / "UPDATE_SESSION_ZERO_CN.md").write_text(
        session_zero_readme, encoding="utf-8", newline="\n"
    )

    files = sorted(
        path
        for path in args.output.rglob("*")
        if path.is_file() and path.name != "deployment_manifest.json"
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_version": "5.0.0",
        "target": "Orange Pi 5B / RK3588 / Ubuntu 22.04 / Python 3.10",
        "acceptance_status": "software_verified_hardware_gated",
        "default_mode": "observe",
        "motor_uart": "/dev/ttyS0 115200 8N1 custom protocol address 1",
        "imu_uart": "/dev/ttyS1 reserved and disabled",
        "rknn_models_included": sorted(copied_models),
        "default_model": config["model"],
        "vision_mode": config["vision"],
        "display_coordinate": config["coordinate"],
        "ready_for_board_software_tests": int8_model.name in copied_models,
        "balance_requires": [
            "track calibration",
            "actuator angle mapping",
            "power-cycle zero persistence or current-boot session zero",
            "identified gains",
            "explicit --arm",
        ],
        "camera_default": config["camera"],
        "roi": config["roi"],
        "files": {
            path.relative_to(args.output).as_posix(): sha256(path) for path in files
        },
    }
    (args.output / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "files"},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
