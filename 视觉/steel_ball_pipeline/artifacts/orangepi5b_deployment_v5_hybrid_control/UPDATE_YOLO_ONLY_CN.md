# YOLO-only 视觉更新

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
