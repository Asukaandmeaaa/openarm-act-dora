# OpenArm ACT + Dora

OpenArm 双臂平台上的独立 ACT 部署工程：固定上游版本、同目标时刻时间融合、锁定动作块、采集一致的控制参数，以及可追溯的运行日志。

**当前配置：预测 100 步，融合后锁定执行 50 步。** 使用实物专家轨迹与仿真普通/纠正轨迹混合更新的 ACT 模型。仓库提供部署代码、模型与处理器下载、MuJoCo 场景、离线测试和复现脚本；不依赖原作者的其它工作空间。

This is an independent OpenArm ACT deployment built on the official Dora components and LeRobot ACT. It includes a pinned checkpoint, reproducible software setup, immutable 50-step commitments from 100-step predictions, software tests, and experiment provenance. See [reproduction details](docs/REPRODUCING.md) for scope and limitations.

## 在另一台电脑复现：推荐 Docker

目标平台：**Linux x86_64**。镜像固定 Ubuntu 24.04 / Python 3.12 基础环境。CPU 即可完成下面的软件测试，首次安装需要网络和数 GB 磁盘空间。Docker 中不启动 ROS 控制节点，不挂载机器人设备。

```bash
git clone https://github.com/Asukaandmeaaa/openarm-act-dora.git
cd openarm-act-dora
docker build -t openarm-act-dora .
docker run --rm -v openarm-act-models:/opt/openarm/models openarm-act-dora
```

最后一条命令会自动下载约 192 MB 的模型附件并校验 SHA256，检查源码/依赖/处理器，运行离线测试，然后用合成的双相机输入执行真实模型推理。模型缓存保存在 Docker volume 中。**这组命令不会打开 CAN、启用电机、打开相机或启动实物 dataflow。**

构建阶段需要 GitHub、PyPI、PyTorch 和 Ubuntu/ROS 软件源可访问。运行测试时首次需要下载 GitHub Release 模型，已有缓存后可以断网运行：

```bash
docker run --rm --network none -v openarm-act-models:/opt/openarm/models openarm-act-dora
```

## 本机安装与 GPU 推理

支持 Ubuntu 24.04 x86_64、Python 3.12。需要已配置 ROS Jazzy 软件源，安装脚本仅使用 KDL/URDF 数学库。详见 [安装与复现](docs/REPRODUCING.md)。

```bash
./scripts/install_system_deps.sh
./scripts/setup.sh cu128         # NVIDIA CUDA 12.8 构建；CPU 用 cpu
python3 scripts/fetch_model.py
./scripts/validate.sh
```

CPU 验证模型能加载和计算；实际 30 Hz 在线预测需要足够快的 GPU，并在目标电脑上验证延迟。新机器重新编译 CAN Python 扩展和重力库，生成自己的 `runtime.local.json`；不会使用原电脑的 Conda、绝对路径或二进制哈希。

## 当前执行方式

| 项目 | 当前值 |
| --- | --- |
| 模型输入 | 右臂 8 轴 + 左臂 8 轴；顶视与右腕 RGB，480×640 |
| 模型输出 | 每次 100×16 个动作值 |
| 预测/目标时间基准 | 30 Hz，固定一个时隙的响应截止时间 |
| 时间融合 | 官方 LeRobot ACTTemporalEnsembler，系数 0，同一目标时刻的历史预测等权 |
| 执行承诺 | 每次锁定融合后的 50 步，后台新预测不改写已锁定块 |
| 最新预测占比 | 满历史块首 1%，块尾约 1.96%；启动/重置后随实际历史数量归一化 |
| 输出处理 | 30 Hz 目标保持 + 官方 250 Hz / 15 Hz 低通 |
| 实物控制 | 专家采集 Kp/Kd；右臂参考速度 + G(q反馈)×1.0；左臂保持+重力；夹爪位置控制 |

模型 SHA256：`2c64de81ba1646175f30c827be1ab6d6166cd0b4652cc8bd1a53408d9f9f85aa`。

这不是全程固定“新 1% / 旧 99%”，也不混合两个不同模型。它融合的是同一个模型在不同时刻对同一目标时刻的预测。

## 仿真与实物入口

- [实物操作说明](docs/HARDWARE.md)：需要先填写本机相机标识，核对 CAN、机械配置和标定；所有启动由操作员执行。
- `./scripts/simctl test`：独立 MuJoCo 图，近似动力学，无真实机器人/相机。
- `./scripts/simctl view`：有桌面显示环境时查看 MuJoCo；不是采集轨迹的逐帧回放。
- `scripts/smoke_dataflow.py`：真实模型 + 假相机/假电机的集成检查，建议 GPU；见复现说明。


## 仓库内容与来源

```text
deployment/           独立部署、时间对齐、锁定块、日志与操作员门控
assets/               采集控制参数、URDF、重力数学源码和 MuJoCo 资源
patches/              固定上游提交之上的完整补丁
sources.lock.json     上游提交、核心依赖、ACT 算法文件哈希
requirements/         Python 直接依赖与完整版本锁
artifacts/model.json  模型下载地址、归档及六个运行文件的 SHA256
scripts/              安装、下载、检查、假节点集成验证、操作入口
tests/                无硬件测试与小型数值夹具
docs/                 复现、模型说明、实验结果和第三方来源
```

原始专家/仿真数据集、训练环境、完整实物日志和原电脑备份未随仓库发布，因此本版本支持**给定模型的部署与软件验证复现**，不宣称从零训练复现。详见 [模型说明](docs/MODEL_CARD.md)。

原创部分使用 [MIT](LICENSE)。上游 Dora/OpenArm、MuJoCo 资源及其它第三方内容保留各自许可证；见 [NOTICE](NOTICE)、[第三方说明](docs/THIRD_PARTY.md) 和 [移植记录](PORTING_NOTES.txt)。本项目不是 Enactic 或 Hugging Face 的官方发布。
