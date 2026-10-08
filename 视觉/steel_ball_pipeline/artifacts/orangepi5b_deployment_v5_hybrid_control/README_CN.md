# Orange Pi 5B 钢球 YOLO 检测与一维坐标控制 v5

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
