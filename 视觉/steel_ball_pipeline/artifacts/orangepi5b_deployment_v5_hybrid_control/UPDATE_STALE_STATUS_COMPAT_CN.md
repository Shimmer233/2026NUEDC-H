# PD42S1 失能状态位兼容更新

适用条件：驱动器收到 `C5 01 FA 01 C1 5C` 后返回成功 ACK，电机已经人工确认释放保持力，但 `0x31` 的使能状态位仍错误地保持为 0。

此兼容模式不会跳过串口帧边界、地址、功能码、校验和、操作结果或回显值检查。重新使能仍必须通过 `0x31` 状态确认。只有失能操作允许在成功 ACK 后忽略这台驱动器已人工确认失真的状态位。

覆盖更新包后执行：

```bash
cd ~/orangepi5b_deployment_v5_hybrid_control
.venv/bin/python scripts/apply_motor_safety_update.py \
  --confirm-stale-disable-status-physically-verified
```

该命令只修改现有 `config/board.yaml` 中的两个电机字段，不会替换 ROI、曝光或其他配置。然后扶稳机构并再次测试：

```bash
.venv/bin/python scripts/calibrate_actuator.py \
  disable --confirm-supported
```

预期结果：

```json
{
  "disabled_verified": true,
  "verification_basis": "ack_plus_physical_test",
  "enabled_reported_by_0x31": true,
  "status_flag_known_stale": true
}
```

`enabled_reported_by_0x31: true` 是已确认的固件状态位异常，不代表代码再次使能了电机。如果 ACK、校验和、地址或功能码异常，命令仍会失败。
