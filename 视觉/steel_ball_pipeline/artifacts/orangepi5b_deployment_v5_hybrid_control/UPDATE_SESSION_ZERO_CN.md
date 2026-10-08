# 当前启动会话零点更新

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
