# 香橙派 5B 上板步骤

当前模型状态为“工程候选，独立验收已由用户豁免”。先完成检测框和圆心运行，再做厘米标定。

1. 把整个目录复制到香橙派 Ubuntu 22.04。
2. 执行 `chmod +x *.sh && ./install.sh`。
3. 执行 `./list_cameras.sh`，找到 USB 摄像头对应的 `/dev/videoN` 或 `/dev/v4l/by-id/...`。
4. 确认 `models/` 中已有转换得到的 `.rknn`。优先试 INT8，异常时改试 FP16。
5. 运行 `./run_detection_only.sh /dev/videoN models/steel_ball_yolov5n_int8.rknn`。

配置默认使用 1280×720、MJPEG、60 FPS 和 ROI `10,140,1110,260`。如果板端相机模式或位置不同，先修改 `config/board.yaml`，不要直接移动检测框补偿。

检测正常后执行 25 cm 水管中位尺标定。按窗口提示依次点击水管四个角点：左上、左下、右下、右上。

```bash
.venv/bin/python scripts/calibrate_pipe.py --device /dev/videoN --width 1280 --height 720 --fps 60
```

标定文件生成后，去掉 `--detection-only` 运行 `board/steel_ball_tracker.py`，即可输出水管中心线坐标和速度 CSV。`s_cm` 以水管中心为 `0 cm`，左端约为 `-12.5 cm`，右端约为 `+12.5 cm`。

```bash
.venv/bin/python board/steel_ball_tracker.py --config config/board.yaml --camera-device /dev/videoN --model models/steel_ball_yolov5n_int8.rknn
```
