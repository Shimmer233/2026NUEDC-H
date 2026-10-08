# PD42S1 使能状态验证更新

本更新继续使用 `/dev/ttyS0`、115200、8N1、地址 1 和 `C5...5C` 自定义协议，不会切换到 Modbus。

原实现只检查 `0xFA` 应答是否回显请求值。现在每次使能或失能后还会轮询 `0x31` 系统状态，只有实际状态一致才返回成功。串口读取超时默认值也由 0.04 秒调整为 0.2 秒。

覆盖更新包后，先用支架或手扶稳水管，保持物理断电开关可立即触及，然后执行：

```bash
cd ~/orangepi5b_deployment_v5_hybrid_control
.venv/bin/python scripts/apply_motor_safety_update.py
.venv/bin/python scripts/calibrate_actuator.py disable --confirm-supported
```

成功结果必须同时包含：

```json
{
  "disabled_verified": true,
  "enabled": false
}
```

如果出现 `driver acknowledged disabled, but 0x31 still reports enabled`，说明发送帧和应答均正确，但驱动器没有保持失能状态。此时立即使用物理开关断开电机电源，不要执行 `zero`、`jog`、`identify` 或 `balance`。

手册第 4 章的 `C5 01 FA ... 5C` 是自定义协议；第 5 章的 `01 06 00 FA ... CRC16` 是 Modbus。运行程序不能混用两套协议。
