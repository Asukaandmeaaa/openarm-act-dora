# 实物操作：只由现场操作员执行

先阅读根目录 README 和 PORTING_NOTES.txt。离线验证完成后，仍需现场核对机器人型号、零位、方向、限位、CAN 接口、相机视角与训练布置。发布版没有在另一台机器人上完成实物验证。

## 本机配置

```bash
cp assets/cameras.example.json assets/cameras.local.json
cp assets/hardware.yaml assets/hardware.local.yaml
```

填写顶视 RealSense 序列号和右腕 `/dev/v4l/by-id/…` 路径。顶视采集 RGB+depth 640×480@30，使用 RGB；右腕要求 MJPG 640×480@60，再按策略图节拍使用最新帧。局部配置被 Git 忽略，避免把机器标识传给别人。可用 `OPENARM_CAMERA_CONFIG` / `OPENARM_HARDWARE_CONFIG` 指定其它配置路径。

默认 can0=右臂，can1=左臂，模型顺序右8轴再左8轴。硬件 YAML 是原专家采集配置，不能把它视为任意机器人都适用的自动标定。不要为消除跟踪误差自行增加增益或更换重力补偿。

控制方式保持：右臂参考位置/参考速度 + G(q反馈)×1.0；左臂位置保持+重力；夹爪仅位置。模型可能输出左臂动作，但实物图不向左臂转发策略运动。

## 启动顺序

每个终端进入本仓库，使用本仓库 `.venv`。停止其它 ROS、Dora 或 CAN 控制器；空夹爪、支撑双臂、确认回零路径和急停可用。

终端 A：

```bash
unset OPENARM_DEPLOYMENT_CONFIG
./scripts/actctl check
# 开机/重连后需要准备 CAN 时，由操作员运行：
./scripts/actctl prepare-host
./scripts/actctl prepare
./scripts/actctl policy
```

`check` / `prepare` 不访问硬件。`prepare-host` 会检查并准备本机 CAN，可能调用 sudo；policy 只启动模型服务。等待 policy 显示 Listening 后，终端 B：

```bash
./scripts/actctl dora
```

**这一步会打开真实相机与 CAN，并构造机器人节点。** 它不属于离线验证。终端 C：

```bash
./scripts/actctl enable
# 输入 ENABLE；双臂使能并约 5 秒回零，等待过程结束。
./scripts/actctl ready
# 现场确认姿态、布置和双臂状态后：
./scripts/actctl start
```

READY 说明软件启动条件满足，不是机械到位测量。ENABLE/HOME 都会使能并回零；不要连续重复两者。

若提示 `/proc/net/can` 不存在，操作员先执行 `sudo modprobe can_raw`，再检查主机 CAN 设置。注册表仍不可读时应解决主机/网络命名空间问题，不绕过独占检查。不要删除锁文件来强行启动。

## 停止与日志

```bash
./scripts/actctl stop
# 支撑双臂后：
./scripts/actctl disable
# 输入 DISABLE，现场确认停用，再退出 Dora 和 policy。
```

STOP 取消当前块与后续策略目标，保持最后发送位置并更新重力，参考速度归零；它不是断电，也不是急停。日志中的 `disable_attempted` 不等于现场确认已断使能。

每次运行重新 `prepare`。源码、模型、硬件配置发生变化后也需新会话。失败会话的请求不可复用，不删旧日志。

- `results/current`：本机当前会话路径。
- `manifest.json`：源码、模型、处理器及配置 SHA。
- `executor.jsonl`：锁定的50个目标、贡献数、最新权重和逐步执行记录。
- `policy_client.jsonl`：原始100步预测、时隙和迟到情况。
- `observations/index.jsonl`：推理输入保存/丢样记录；Arrow 输入约可达到 3.3 GB/分钟，安排足够磁盘。

源码 Git 回退只恢复程序文件；不要恢复或复用旧 `*.request`。原作者的历史成功备份仍留在原电脑，未把那些物理会话打包到此仓库。
