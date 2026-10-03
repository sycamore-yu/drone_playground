# Drone Playground

用于无人机学习、规划与控制研究的配置驱动仿真平台。通过完整的方法配方和环境预设，组合策略网络、优化器、传感器、任务、控制器和动力学，并保存可追溯的训练记录、冻结评测和物理轨迹回放。

学习方法使用 JAX、Brax 和 Crazyflow；SUPER 与 EGO-Planner 通过项目独立管理的 ROS 容器接入。PointNet/GRU 点云方法保留论文来源与训练配方；LOTF 只提供 high-fidelity 和 simplified 两种可组合动力学。

处理链的组织方式 inspired by [FlightBench 表Ⅲ与图4](https://arxiv.org/abs/2406.05687)：不同方法可覆盖不同区段，通过 Reference、Setpoint 或执行器输入接入后续执行系统。本项目研究 JAX 仿真、可微强化学习与 MPC 的组合。当前为研究预览；[第一版交付规格](docs/notes/archive/release-plan.md)定义18个组合；EGO两格由用户接受交付，其余16格继续质量验收，当前14格按现行规则质量通过（其中深度两格按六个主要场景的新规则），共16/18格满足交付要求。

## 安装与运行

开发环境为 Linux、Python 3.13 和 Pixi。GPU 训练使用 NVIDIA CUDA；依赖版本、源码提交及必要补丁分别固定在 `pixi.lock` 和 `third_party/sources.json`，由 `scripts/tools/setup.py` 负责准备本地源码缓存。

```bash
git clone https://github.com/sycamore-yu/drone_playground.git
cd drone_playground
python3 scripts/tools/setup.py sources
pixi install --locked

# 查看完整解析配置，然后开始独立训练。
pixi run train experiment=control/ppo env=hovering --cfg job
pixi run train experiment=control/ppo env=hovering runtime.device=gpu run_id=ppo-hover-example
```

`experiment` 选择完整实验组合，`env` 选择完整环境。`drone_playground.load()` 直接使用 Hydra 加载环境，不创建 registry，也不隐式选择训练算法。`train` 负责学习，`eval` 使用冻结参数，`play` 执行或查看物理轨迹。已有运行目录受到覆盖保护；新的试验使用新的 `run_id`，已有训练通过检查点恢复。

```bash
# 查看当前选定结果及其原始 run / checkpoint 引用。
cat results/selected/v1-18-cells.json

# 从明确的冻结参数重新评测；如需完整可视化回放再显式开启。
pixi run eval \
  checkpoint=results/runs/racing/ppo/<training-run>/checkpoints/<checkpoint>.pkl \
  runtime.device=gpu evaluation.episodes=32 evaluation.record_replays=true \
  run_id=recheck-racing

# 查看这次评测保存的回放。
pixi run play replay=results/runs/racing/ppo/recheck-racing/rollouts
```

当前配置为 v4，原生 RPC 为 v2。历史 v3 检查点需按[显式迁移步骤](docs/runbook.md#历史检查点)创建新副本；原始权重和历史结果不改写。

权重、轨迹和依赖缓存由本地结果包管理，Git 保存源码、配置、协议和精简证据。新的源码克隆需要自行训练或取得相应结果包。原生规划器的容器准备及完整命令见[操作手册](docs/runbook.md)。

## 冻结的正式验收结果

以下为2026-09-29选定的六方法、五任务历史验收矩阵。控制任务单元为“完成回合／总回合；位置均方根误差（米）”，导航单元为到达次数与终止事件。后续导航调参记录独立归档；本表始终对应[明确的运行选择](artifacts/verification/final-acceptance/selection.json)。

| 方法 | 悬停 | 跟踪 | 竞速 | 静态导航 | 动态导航 |
|---|---|---|---|---|---|
| PPO | 32/32；0.152 | 32/32；0.036 | 32/32；0.015 | 0/6，6次碰撞 | 0/6，6次碰撞 |
| BPTT | 32/32；0.151 | 32/32；0.029 | 32/32；0.007 | 0/6，6次碰撞 | 0/6，6次碰撞 |
| SHAC | 32/32；0.152 | 32/32；0.030 | 32/32；0.010 | 0/6，6次碰撞 | 0/6，6次碰撞 |
| 点云方法¹ | 32/32；0.177 | 32/32；0.076 | 32/32；0.065 | 0/16，15次碰撞、1次超时 | 0/16，14次碰撞、2次超时 |
| SUPER | 2/2；0.275 | 2/2；0.251 | 0/2 | 6/6 | 6/6 |
| EGO-Planner | 2/2；0.367 | 2/2；0.668 | 0/2，2次超时 | 0/6，6次碰撞 | 0/6，6次碰撞 |

¹ 点云控制任务采用显式点坐标尺度0.02的控制迁移配方；导航采用论文公开信息重建方法的第30000次更新冻结参数。两者具有各自的训练配置。SHAC 竞速使用 BPTT 参数热启动后继续执行真实 SHAC 更新。

这是一份平台集成与任务质量记录：控制学习方法各评测32回合，原生规划器控制任务各2回合；PPO、BPTT、SHAC 导航记录对应4096次交互的工程短训练。完成率、误差阈值和预算完成分别报告；SUPER、EGO-Planner 的控制误差仍高于0.25米质量阈值。完整488回合分母、245份回放索引、参数摘要及运行条件见[正式验收说明](artifacts/verification/final-acceptance/README.md)。

## 实验与环境

`experiment=control/ppo`、`control/bptt`、`control/shac` 和 `control/apg` 提供控制学习基线；导航学习使用 `experiment=navigation/ppo|bptt|shac`。点云原始重建为 `experiment=papers/differentiable_pointcloud`，导航适配为 `experiment=navigation/differentiable_pointcloud`；SUPER、EGO-Planner 与 MPC 分别使用 `papers/super`、`papers/ego_planner`、`control/attitude_mpc` 和 `control/sampling_mpc`。LOTF 只作为动力学来源，通过 `dynamics@env.dynamics=lotf_high_fidelity` 或 `lotf_simplified` 组合；diffRL 的反向规则由 `algorithm.gradient.transition` 选择。训练分布与 benchmark 协议属于对应 experiment，不再另设 training/evaluation preset 组；质量状态见[待办](docs/status.md)。

[架构与接口](docs/architecture.md)说明六个环境组件、Reference/Setpoint、训练与运行方法、场景资产和评测的职责。悬停、跟踪、竞速和导航分别使用现有任务配置；Navigation 的静态与动态变化由场景集合表达。论文原始任务与项目迁移分别标记。

核心环境为 `hovering`、`tracking`、`racing`、`navigation/static` 和 `navigation/dynamic`。Navigation 的八张固定场景包括 S01/S02/S03/S06 与 D01/D02/D03/D06，共用100×40米几何、96米起终点距离、0.5米到达半径，以及导航第二版的300秒时限和20米/秒名义速度上限。实际命令速度与达到的速度另行记录。

## 文档与开发

[架构与数据流](docs/architecture.md)说明组件职责；[操作手册](docs/runbook.md)提供安装、训练、评测与回放命令；[目录职责](docs/directory-layout.md)用于定位现役源码与原生部署；[开发约定](docs/development.md)统一测试、设备选择和产物管理；[状态](docs/status.md)只保留现役状态。

```bash
JAX_PLATFORMS=cpu pixi run test tests/integration/configuration/test_direct_hydra_environment.py
pixi run lint
```

历史实验编排与一次性交付核验脚本归档在 `tmp/retired-experiments/`，不属于现役平台入口。完整 MPC 数值回归需要先执行 `pixi run setup-acados`；具体条件见[操作手册](docs/runbook.md)。正式训练默认使用 GPU；CPU 用于轻量验证、测试及原生规划器宿主执行。

## 许可与来源

项目使用 [GPL-3.0-only](LICENSE)。各上游代码、移植实现、补丁与外部进程的来源见[第三方声明](docs/licenses/third_party.md)；固定来源由 `third_party/sources.json` 声明，并由 `scripts/tools/setup.py` 准备本地缓存。公开源代码与研究结果的适用范围，以具体方法配置和评测协议为准。

当前职责与命名见[目录说明](docs/directory-layout.md)。`docs/notes/archive/` 只保存历史设计与交付快照，不作为现役源码地图。完整组合统一使用 `experiment=...`，旧 `method=...` 入口已迁移。
