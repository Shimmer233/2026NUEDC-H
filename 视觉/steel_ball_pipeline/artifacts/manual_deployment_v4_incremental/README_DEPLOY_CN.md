# 钢球 YOLO v4 手动转换与部署

本包用于在香橙派上生成 RKNN、旁路测试并手动启用新模型。旧模型不会在解压、转换、暂存或测试时被覆盖。只有本次暂存模型的 `normal`、`fast`、`empty` 三项测试全部通过，并明确执行 `activate_model.sh --confirm` 后，活动模型才会被替换。

重要状态：包内已经包含训练完成的 ONNX、300 张新旧场景混合量化图片和 RKNN-Toolkit2 2.3.2 ARM64 wheel，但不包含尚未生成的 v4 RKNN。必须先执行第 3 节转换。

## 1. 复制到香橙派

在 Windows PowerShell 中把最终 ZIP 传到香橙派。将用户名、IP 和目标目录按实际情况替换：

```powershell
scp "steel_ball_v4_manual_deployment.zip" orangepi@192.168.1.100:/home/orangepi/
```

在香橙派终端执行：

```bash
cd /home/orangepi
unzip steel_ball_v4_manual_deployment.zip
cd /home/orangepi/steel_ball_v4_manual_deployment
sha256sum -c PACKAGE_SHA256SUMS.txt
unzip rknn_conversion_bundle_v4_incremental.zip -d /home/orangepi
```

全部条目必须显示 `OK`。设置后续命令使用的路径：

```bash
PACKAGE_DIR=/home/orangepi/steel_ball_v4_manual_deployment
PROJECT_DIR=/home/orangepi/orangepi5b_vision_only
BUNDLE_DIR=/home/orangepi/rknn_conversion_bundle_v4_incremental
CAMERA=/dev/video0
```

如果工程或用户名不同，只修改这四个变量。

## 2. 确认两套 Python 环境

正式工程的 `.venv` 用于运行 RKNNLite，不能拿它转换模型。转换必须使用能够导入 `rknn.api.RKNN` 的完整 `rknn-toolkit2==2.3.2` 环境。

已有转换环境时，把 `CONVERT_PY` 指向该环境的 Python：

```bash
CONVERT_PY=/实际的/rknn-toolkit2-2.3.2环境/bin/python
"$CONVERT_PY" --version
"$CONVERT_PY" -c 'from importlib.metadata import version; from rknn.api import RKNN; print(version("rknn-toolkit2"))'
"$PROJECT_DIR/.venv/bin/python" -c 'from rknnlite.api import RKNNLite; print("RKNNLite import OK")'
```

第二条命令必须输出 `2.3.2`。只有 `rknn-toolkit2-v2.3.2` 源码目录或 wheel 文件，并不代表完整环境已经安装；导入失败时使用下一节的“自建环境”命令。

## 3. 生成 FP16 和 INT8 RKNN

先验证转换包：

```bash
cd "$BUNDLE_DIR"
python3.10 verify_bundle.py
```

必须看到 `bundle verified: 300 calibration images`。

使用已经安装好的 Toolkit2 2.3.2 环境：

```bash
cd "$BUNDLE_DIR"
chmod +x *.sh
set -o pipefail
RKNN_PYTHON="$CONVERT_PY" ./run_conversion_existing_env.sh 2>&1 | tee conversion.log
test "${PIPESTATUS[0]}" -eq 0
sha256sum -c output/SHA256SUMS.txt
```

没有可用环境时，在 ARM64 香橙派上自建 Python 3.10 环境。脚本默认使用清华 PyPI 镜像，并设置 120 秒超时和 10 次重试：

```bash
cd "$BUNDLE_DIR"
chmod +x *.sh
set -o pipefail
./run_conversion_arm64.sh 2>&1 | tee conversion.log
test "${PIPESTATUS[0]}" -eq 0
sha256sum -c output/SHA256SUMS.txt
```

清华源不可用时，可切换阿里云源后重跑；已有的部分下载会被 pip 缓存复用：

```bash
RKNN_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
  ./run_conversion_arm64.sh 2>&1 | tee conversion.log
```

需要恢复官方源时：

```bash
RKNN_PIP_INDEX_URL=https://pypi.org/simple \
  ./run_conversion_arm64.sh 2>&1 | tee conversion.log
```

成功后应有以下文件：

```text
output/steel_ball_yolov5n_fp16.rknn
output/steel_ball_yolov5n_int8.rknn
output/steel_ball_yolov5n_fp16.json
output/steel_ball_yolov5n_int8.json
output/SHA256SUMS.txt
```

## 4. 暂存新模型

```bash
cd "$PACKAGE_DIR"
chmod +x scripts/*.sh
./scripts/stage_model.sh "$PROJECT_DIR" "$BUNDLE_DIR"
```

脚本会验证 RKNN 来自本次 ONNX、目标是 RK3588、Toolkit 版本为 2.3.2；INT8 还必须使用本包批准的 300 张量化清单和指定归一化参数。通过后模型才会复制为版本化文件。测试配置从板上当前 `vision.yaml` 复制，只修改模型路径、测试输出路径和以下 ROI：

```yaml
roi:
  x: 10
  y: 150
  width: 600
  height: 105
  output_width: 470
  output_height: 110
```

不会修改 `track_geometry.yaml`、`track_calibration.yaml`、相机参数、检测阈值或当前活动模型路径。

## 5. 停止当前程序

测试、启用和回滚脚本都会同时检查视觉进程和摄像头占用；检测到占用时会拒绝继续。先查看并正常结束实际进程：

```bash
pgrep -af 'board/steel_ball_vision.py|examples/read_position_velocity.py'
sudo fuser -v "$CAMERA"
kill -INT <上一步确认的PID>
pgrep -af 'board/steel_ball_vision.py|examples/read_position_velocity.py'
sudo fuser -v "$CAMERA"
```

不要使用 `pkill python` 或 `fuser -k`。若进程自动重启，使用 `ps -o pid,ppid,cmd -p <PID>` 找到实际管理它的服务或桌面启动项，先停止该管理程序。

## 6. 三项旁路测试

每次先布置好场景，再执行对应命令：

```bash
cd "$PACKAGE_DIR"

# 正常移动；过程中在旧模型易丢失的静止位置停球
./scripts/run_staged_test.sh "$PROJECT_DIR" normal 60 "$CAMERA"

# 快速移动；覆盖运动拖影和左右挡板碰撞
./scripts/run_staged_test.sh "$PROJECT_DIR" fast 60 "$CAMERA"

# 完全移除钢球，保留真实空轨道和正常手部干扰
./scripts/run_staged_test.sh "$PROJECT_DIR" empty 60 "$CAMERA"
```

每轮都会保存唯一目录、`run.log`、轨迹 CSV、性能 JSON 和 `summary.json`。日志出现 `Traceback`、`YOLO inference warning` 或 `RuntimeError` 会直接判失败。

三项测试均要求融合速率中位数和整段平均速率都不低于 30 FPS。有球测试同时要求置信度和 `s_cm` 位置输出的可用率至少 99%，各自连续空缺不超过 3 个融合采样点，并要求跟踪器接受的新测量不低于 30 Hz。现有工程 CSV 没有单独记录“新推理完成”标志，因此报告中的 `recall` 是融合采样代理指标，不应当作独立数据集精度。空轨道要求原始候选不超过 1/1000 帧，并且 `detected`、`predicted` 和 `s_cm` 输出必须全部为零或空。

某轮失败时，重新检查场景或模型后重跑该项即可；启用门禁只认本次暂存模型、同一摄像头设备下每个场景的最新一次测试。
如果暂存后修改了板上的 `vision.yaml`、`track_geometry.yaml` 或 `track_calibration.yaml`，哈希门禁会关闭；必须重新暂存并重做三项短测试。

## 7. 正式启用

三项测试都通过后执行：

```bash
cd "$PACKAGE_DIR"
./scripts/activate_model.sh "$PROJECT_DIR" --confirm "$CAMERA"
```

启用脚本会：

- 在板上完整备份当前 `models/` 和 `config/`，逐文件记录并复核 SHA-256，然后在替换前发布 `LAST_V4_BACKUP` 回滚指针；
- 再次验证暂存模型、配置和三份测试属于同一次暂存；
- 逐文件原子替换两套 RKNN、两个转换 JSON、模型校验文件和 `vision.yaml`，活动 INT8 文件最后提交；
- 保持活动路径 `models/steel_ball_yolov5n_int8.rknn` 不变；
- 执行 10 秒唯一目录冒烟测试；冒烟失败时自动恢复完整旧备份。

如果启用期间断电或被强制终止，不要直接启动正式程序；先按第 8 节使用已经提前写好的 `LAST_V4_BACKUP` 完整回滚。

成功后按原方式启动正式程序：

```bash
cd "$PROJECT_DIR"
./run_vision.sh "$CAMERA"
```

原香橙派虚拟环境、解码器和三输出接口不需要改变。

## 8. 回滚

先正常停止正在运行的程序。默认回滚到最近一次 v4 启用前备份：

```bash
cd "$PACKAGE_DIR"
STEEL_BALL_CAMERA="$CAMERA" ./scripts/rollback_model.sh "$PROJECT_DIR" --confirm
```

指定某个备份和摄像头时：

```bash
./scripts/rollback_model.sh \
  "$PROJECT_DIR" \
  --confirm \
  "$PROJECT_DIR/backups/pre_v4_YYYYMMDDTHHMMSSZ" \
  "$CAMERA"
```

回滚会恢复备份中的完整 `models/` 和 `config/`，包括旧 `vision.yaml`、轨道几何和厘米标定，并按 manifest 逐文件复核模型、JSON 和全部配置，然后执行 10 秒冒烟测试。回滚前的 v4 文件会另外保存在 `backups/post_v4_before_rollback_*` 并生成独立 manifest；回滚过程失败时脚本会验证并恢复这份状态。
