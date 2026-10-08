# Orange Pi 5B 钢球视觉识别（纯视觉版）

本目录从 `orangepi5b_deployment_v5_hybrid_control` 中独立抽取，只包含相机、轨道标定、RKNN YOLO 检测、位置/速度估计、显示、CSV 输出和可选 UART 发送。默认运行不会导入电机驱动，也不会打开串口；只有 API 示例显式传入 `--uart-device` 时才会打开对应 UART。

## 环境

- Orange Pi 5B / RK3588
- Ubuntu 22.04 / Python 3.10
- 640x480 MJPG USB 相机
- 随目录提供的 RKNNLite 2.3.2 aarch64 wheel

## 安装与运行

```bash
cd orangepi5b_vision_only
chmod +x *.sh
./install.sh
./list_cameras.sh
./run_vision.sh /dev/video0
```

按 `Q` 或 `Esc` 退出。无显示环境可运行：

```bash
./run_vision.sh /dev/video0 --headless --duration 60
```

识别结果写入 `runs/vision/trajectory.csv`，性能统计写入 `runs/vision/trajectory.performance.json`。CSV 中 `s_cm` 是从轨道左端起算的 `0..25 cm` 坐标；画面显示以轨道中心为零点，左负右正。

## 在后续 Python 程序中读取位置和速度

不要同时运行 `run_vision.sh`，因为摄像头只能由一个进程占用。直接在后续程序中导入 `SteelBallVision`：

```python
from steel_ball.realtime import SteelBallVision

last_frame_id = -1
with SteelBallVision(camera_device="/dev/video0") as vision:
    while True:
        state = vision.wait_for_update(last_frame_id, timeout=1.0)
        if state is None:
            continue
        last_frame_id = state.frame_id

        if not state.valid:
            continue

        ball_position_cm = state.centered_position_cm
        ball_velocity_cm_s = state.velocity_cm_s

        # 在这里调用后续控制或通信程序。
        print(ball_position_cm, ball_velocity_cm_s)
```

也可以随时非阻塞读取 `state = vision.latest`。变量定义如下：

- `state.centered_position_cm` / `state.x_cm`：以轨道中心为 `0`，左负右正，单位 cm。
- `state.position_cm`：以轨道左端为 `0`，范围约 `0..25`，单位 cm。
- `state.velocity_cm_s` / `state.v_cm_s`：向右为正，单位 cm/s。
- `state.valid`：位置和速度当前是否可用；后续控制程序必须先检查它。
- `state.detected`：本次状态来自新的 YOLO 测量。
- `state.predicted`：本次状态是短时卡尔曼预测，未直接看到球。

可先运行完整示例：

```bash
chmod +x run_api_example.sh
./run_api_example.sh /dev/video0
```

如果需要在 API 模式下把位置和速度通过香橙派 UART1 发给下位机，可运行：

```bash
./run_api_example.sh /dev/video0 --uart-device /dev/ttyS1 --uart-baud 115200
```

如果需要同时打开触摸屏图形界面，可加 `--gui`：

```bash
./run_api_example.sh /dev/video0 --gui --uart-device /dev/ttyS1 --uart-baud 115200
```

界面中有两个触摸按钮：

- `MID 0`：目标点设为轨道中点，也就是原来的 `0.000 cm`。
- `CUSTOM`：点一下该按钮后，再触摸画面中的任意横向位置；程序会按该触摸点的竖线与轨道标定线相交处换算目标坐标。

启用 `--gui` 后，打印和串口发送的位置会变成 `state.centered_position_cm - target_cm`，也就是把当前目标点重新定义为新的 `0`。速度仍然保持原方向定义，向右为正。

串口只发送 `state.valid == True` 的有效数据，每帧一行 ASCII：

```text
PV,<位置cm>,<速度cm/s>\n
```

例如：

```text
PV,-1.235,+7.890
```

其中中心位置仍然是 `state.centered_position_cm`，以轨道中心为 `0`，左负右正；速度向右为正。下位机按换行分包、按逗号拆字段即可。

注意：未启用 `--gui` 时，`PV` 中的位置就是原始中心坐标；启用 `--gui` 后，`PV` 中的位置是相对当前目标点的新坐标。

## 标定

交付目录保留了原项目当前的相机轨道几何与厘米标定。相机或安装位置变化后应重新标定：

```bash
.venv/bin/python scripts/setup_camera.py --config config/vision.yaml --device /dev/video0
.venv/bin/python scripts/calibrate_track_geometry.py --config config/vision.yaml --device /dev/video0
.venv/bin/python scripts/calibrate_track.py --device /dev/video0 --width 640 --height 480 --fps 120
```

## 目录说明

```text
board/steel_ball_vision.py       纯视觉主程序
steel_ball/detector.py           RKNN YOLO 异步检测与轨道透视
steel_ball/vision.py             图像预处理与圆心精修
steel_ball/rknn_postprocess.py   YOLOv5 输出解码/NMS
steel_ball/tracking.py           一维位置速度卡尔曼估计
steel_ball/calibration.py        像素到厘米标定
steel_ball/camera_controls.py    V4L2 相机参数
steel_ball/realtime.py           可导入的位置/速度实时 API
examples/                        API 调用示例
scripts/                         相机和轨道标定工具
config/vision.yaml               纯视觉配置
models/                          INT8/FP16 RKNN 模型
```

该目录刻意不包含 `pd42s1.py`、`control.py`、`session_zero.py`、电机标定配置、控制参数、控制日志和电机启动脚本。
