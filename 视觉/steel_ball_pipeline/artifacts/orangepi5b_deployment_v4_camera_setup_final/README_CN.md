# 香橙派 5B 上板步骤

当前模型状态为“工程候选，独立验收已由用户豁免”。先完成检测框和圆心运行，再做厘米标定。

1. 把整个目录复制到香橙派 Ubuntu 22.04。
2. 执行 `chmod +x *.sh && ./install.sh`。
3. 执行 `./list_cameras.sh`，找到 USB 摄像头对应的 `/dev/videoN` 或 `/dev/v4l/by-id/...`。
4. 确认 `models/` 中已有转换得到的 `.rknn`。优先试 INT8，异常时改试 FP16。
5. 运行 `./run_detection_only.sh /dev/videoN models/steel_ball_yolov5n_int8.rknn`。

配置默认使用 640×480、MJPEG、120 FPS 和 ROI `10,150,555,105`。摄像头采集尺寸是 640×480，模型静态输入仍为 640×640；程序会自动裁剪轨道 ROI 并补边。

首次运行前建议执行摄像头设置工具，用鼠标框住完整白色轨道并自动调整曝光：

```bash
.venv/bin/python scripts/setup_camera.py --device /dev/videoN
```

工具将曝光限制在 120 FPS 帧周期以内，亮度不足时再增加增益，并把 ROI 与固定控制值保存到 `config/board.yaml`。

检测正常后执行标定：

```bash
.venv/bin/python scripts/calibrate_track.py --device /dev/videoN --width 640 --height 480 --fps 120
```

标定文件生成后，去掉 `--detection-only` 运行 `board/steel_ball_tracker.py`，即可输出厘米位置和速度 CSV。
