# 钢球 YOLOv5 训练与 RK3588 部署

本项目与旧的 `yolov5-master` 训练产物完全隔离。旧 `best.pt`、旧 ONNX、旧标签、缓存和划分均不读取；初始化权重只允许来自 Ultralytics YOLOv5 v7.0 官方发布页，并在训练前核对下载清单和 SHA-256。

## 当前数据状态

- 原始图片：`D:\University\NUEDC\yolov5-master\yolov5-master\VOCData\images`，1174 张，脚本只读。
- 独立数据集：`D:\University\NUEDC\datasets\steel_ball_v2`。
- 固定 ROI：`x=145, y=190, width=470, height=110`。
- 时间块划分：训练 839 张（21:05–21:11），验证 335 张（21:12–21:13），测试集等待独立视频。
- OpenCV 自动框结构有效 1174/1174；75 张被标为不确定。所有 1174 张仍是 `pending`，训练门禁因此处于关闭状态。

## 1. 人工复核与批准

在项目目录运行：

```powershell
cd D:\University\NUEDC\steel_ball_pipeline
python scripts\review_labels.py --pending-only
```

左键拖框；`Enter` 或空格接受并前进；没有球时按 `N` 保存为空标签并前进；`A/D` 前后移动；`R` 恢复自动框；`Q` 保存并退出。必须逐张确认。完成后执行：

```powershell
python scripts\audit_dataset.py --no-render
python scripts\approve_dataset.py
python scripts\approve_dataset.py --confirm
```

批准文件记录全部训练/验证标签的 SHA-256；批准后改动任一标签都会重新阻止训练。

## 2. 训练 YOLOv5n

官方 `yolov5n.pt` 已下载到 `weights`，来源和哈希在同名 JSON 中。默认训练 200 epochs、patience 30、seed 42、640×640、batch 32，并启用 YOLOv5 AutoAnchor。Mosaic、MixUp、翻转和透视均关闭。训练脚本会生成 40% 的确定性水平运动模糊训练副本，验证集不增强。

```powershell
python scripts\train_model.py --model yolov5n
```

输出位于 `runs\train\steel_ball_yolov5n_seed42_*`；命令、环境、依赖、Git 提交、数据批准哈希、增强清单和随机种子位于 `runs\manifests`。

YOLOv5s 只有在 YOLOv5n 独立测试召回低于 99% 时才能运行：

```powershell
python scripts\download_official_weights.py yolov5s
python scripts\train_model.py --model yolov5s --yolov5n-test-report runs\reports\yolov5n_test.json
```

## 3. 独立测试集

分别录制正常往返、高速和空轨道视频，原始视频完整保存在 `test_sources`：

```powershell
python scripts\capture_test_video.py --kind normal --seconds 30
python scripts\capture_test_video.py --kind fast --seconds 30
python scripts\capture_test_video.py --kind empty --seconds 30
```

导入有球视频会再次使用纯 OpenCV 预标注并要求人工复核；空轨道必须先完整观看，再显式确认：

```powershell
python scripts\import_test_video.py --video <normal.avi> --kind ball --prefix normal
python scripts\import_test_video.py --video <fast.avi> --kind ball --prefix fast
python scripts\import_test_video.py --video <empty.avi> --kind empty --prefix empty --confirm-empty
python scripts\review_labels.py --pending-only
python scripts\evaluate_model.py --weights <best.pt>
```

将评估生成的 `labels` 目录交给指标脚本；它报告总召回、暗帧、强反光、轨道两端、高模糊、最长连续漏检和空轨道误检：

```powershell
python scripts\analyze_predictions.py --dataset D:\University\NUEDC\datasets\steel_ball_v2 --split test --predictions <labels目录> --output runs\reports\yolov5n_test.json
```

## 4. ONNX 与 RKNN

导出会调用干净 Rockchip 分支的 `--rknpu` 模式，得到三个不含后处理的检测头：

```powershell
python scripts\export_rknn_onnx.py --weights <best.pt> --output artifacts\steel_ball_yolov5n.onnx
python scripts\make_quantization_list.py --count 300
```

RKNN-Toolkit2 2.3.2 转换应在 Rockchip 支持的 x86_64 Linux/Python 环境运行。安装官方 2.3.2 wheel 后分别执行：

```bash
python scripts/convert_rknn.py --onnx artifacts/steel_ball_yolov5n.onnx --mode fp16 --output artifacts/steel_ball_yolov5n_fp16.rknn
python scripts/convert_rknn.py --onnx artifacts/steel_ball_yolov5n.onnx --mode int8 --dataset-list artifacts/quantization.txt --output artifacts/steel_ball_yolov5n_int8.rknn
```

在香橙派上分别运行完整测试集并分析：

```bash
python scripts/infer_dataset.py --backend rknn-lite --model artifacts/steel_ball_yolov5n_fp16.rknn --dataset /path/to/steel_ball_v2 --split test --output runs/rknn_fp16
python scripts/infer_dataset.py --backend rknn-lite --model artifacts/steel_ball_yolov5n_int8.rknn --dataset /path/to/steel_ball_v2 --split test --output runs/rknn_int8
python scripts/compare_rknn_reports.py --fp16 runs/reports/fp16.json --int8 runs/reports/int8.json --output runs/reports/rknn_choice.json
```

INT8 相对 FP16 下降不超过 0.5 个百分点时选择 INT8，否则选择 FP16。

## 5. 标定与板端追踪

把钢球依次放在至少 10 个已知位置，点击球心生成标定文件：

```bash
python scripts/calibrate_track.py --positions 0,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25
```

修改 `config/board.yaml` 中模型和标定路径后，安装 `board/requirements-board.txt` 以及 Rockchip RK3588/aarch64 的 RKNNLite2 2.3.2 wheel，然后运行：

```bash
python board/steel_ball_tracker.py --config config/board.yaml
python board/steel_ball_tracker.py --config config/board.yaml --headless --duration 600
```

采集线程只保留最新帧，不会积累队列。程序使用检测框内圆/椭圆拟合精修中心，一维 `[位置, 速度]` 卡尔曼滤波最多补偿 3 帧；CSV 字段固定，10 分钟测试另写 `trajectory.performance.json`，包含中位 FPS、P95 延迟、跳帧和内存增长。
