# Drone Playground

用于无人机学习、规划与控制研究的配置驱动仿真平台。通过完整的方法配方和环境预设，组合策略网络、优化器、传感器、任务、控制器和动力学，并保存可追溯的训练记录、冻结评测和物理轨迹回放。

学习方法使用 JAX、Brax 和 Crazyflow；SUPER 与 EGO-Planner 通过项目独立管理的 ROS 容器接入。PointNet/GRU 点云方法与 LOTF 分别保留自己的来源、动力学和训练身份。

处理链的组织方式 inspired by [FlightBench 表Ⅲ与图4](https://arxiv.org/abs/2406.05687)：不同方法可覆盖不同区段，在轨迹、航点或运动命令处接入后续执行系统。本项目研究 JAX 仿真、可微强化学习与 MPC 的组合。当前为研究预览；[第一版交付规格](docs/release-plan.md)定义18个组合；EGO两格由用户接受交付，其余16格继续质量验收，当前14格按现行规则质量通过（其中深度两格按六个主要场景的新规则），共16/18格满足交付要求。

## 安装与运行

开发环境为 Linux、Python 3.13 和 Pixi。GPU 训练使用 NVIDIA CUDA；依赖版本、源码提交及必要补丁分别固定在 `pixi.lock` 和 `third_party/sources.yaml`。

```bash
git clone https://github.com/sycamore-yu/drone_playground.git
cd drone_playground
python3 scripts/tools/fetch_sources.py
pixi install --locked

# 查看完整解析配置，然后开始独立训练。
pixi run train method=learning/ppo env=hovering --cfg job
pixi run train method=learning/ppo env=hovering runtime.device=gpu run_id=ppo-hover-example
```

`method` 选择完整方法，`env` 选择完整环境。`train` 负责学习，`eval` 使用冻结参数，`play` 执行或查看物理轨迹。已有运行目录受到覆盖保护；新的试验使用新的 `run_id`，已有训练通过检查点恢复。

```bash
# 安装正式结果包后，查看已保存的 PPO 竞速轨迹。
pixi run play replay=experiments/final-acceptance-heldout-ppo-racing-v1/rollouts

# 从明确的冻结参数重新评测，输出至独立运行目录。
pixi run eval \
  checkpoint=experiments/final-acceptance-ppo-racing-t0/checkpoints/step-0002097152.pkl \
  runtime.device=gpu evaluation.split=heldout evaluation.episodes=32 \
  run_id=recheck-ppo-racing
```

权重、轨迹和依赖缓存由本地结果包管理，Git 保存源码、配置、协议和精简证据。新的源码克隆需要自行训练或取得相应结果包。原生规划器的容器准备及完整命令见[操作手册](docs/runbook.md)。

## 冻结的正式验收结果

以下为2026-09-29选定的六方法、五任务历史验收矩阵。控制任务单元为“完成回合／总回合；位置均方根误差（米）”，导航单元为到达次数与终止事件。后续导航调参记录独立归档；本表始终对应[明确的运行选择](docs/verification/final-acceptance/selection.json)。

| 方法 | 悬停 | 跟踪 | 竞速 | 静态导航 | 动态导航 |
|---|---|---|---|---|---|
| PPO | 32/32；0.152 | 32/32；0.036 | 32/32；0.015 | 0/6，6次碰撞 | 0/6，6次碰撞 |
| BPTT | 32/32；0.151 | 32/32；0.029 | 32/32；0.007 | 0/6，6次碰撞 | 0/6，6次碰撞 |
| SHAC | 32/32；0.152 | 32/32；0.030 | 32/32；0.010 | 0/6，6次碰撞 | 0/6，6次碰撞 |
| 点云方法¹ | 32/32；0.177 | 32/32；0.076 | 32/32；0.065 | 0/16，15次碰撞、1次超时 | 0/16，14次碰撞、2次超时 |
| SUPER | 2/2；0.275 | 2/2；0.251 | 0/2 | 6/6 | 6/6 |
| EGO-Planner | 2/2；0.367 | 2/2；0.668 | 0/2，2次超时 | 0/6，6次碰撞 | 0/6，6次碰撞 |

¹ 点云控制任务采用显式点坐标尺度0.02的控制迁移配方；导航采用论文公开信息重建方法的第30000次更新冻结参数。两者具有各自的训练配置。SHAC 竞速使用 BPTT 参数热启动后继续执行真实 SHAC 更新。

这是一份平台集成与任务质量记录：控制学习方法各评测32回合，原生规划器控制任务各2回合；PPO、BPTT、SHAC 导航记录对应4096次交互的工程短训练。完成率、误差阈值和预算完成分别报告；SUPER、EGO-Planner 的控制误差仍高于0.25米质量阈值。完整488回合分母、245份回放索引、参数摘要及运行条件见[正式验收说明](docs/verification/final-acceptance/README.md)。

## 方法与环境

`learning/ppo`、`learning/bptt`、`learning/shac` 提供通用学习基线；`paper/pointcloud_flight` 与 `paper/lotf` 提供独立的文献方法；`paper/super`、`paper/ego_planner` 和 `optimization/attitude_mpc`、`optimization/sampling_mpc` 提供规划或优化基线。具名导航适配配方单独管理实验性训练变化，质量状态见[待办](docs/backlog.md)。

[项目主表与评测分类](docs/methods.md)按FlightBench的接口组织方式增加传感器和任务列：核心任务族为悬停、跟踪、竞速和导航（静态／动态），着陆与集群协同作为专项扩展，走廊轨迹生成作为组件评测。论文原始任务与项目迁移分别标记。

核心环境为 `hovering`、`tracking`、`racing`、`navigation/static` 和 `navigation/dynamic`。Navigation 的八张固定场景包括 S01/S02/S03/S06 与 D01/D02/D03/D06，共用100×40米几何、96米起终点距离、0.5米到达半径，以及导航第二版的300秒时限和20米/秒名义速度上限。实际命令速度与达到的速度另行记录。

## 文档与开发

[架构与数据流](docs/architecture.md)说明组件职责；[操作手册](docs/runbook.md)提供安装、训练、评测与回放命令；[评测协议](docs/evaluation.md)定义指标和信息边界；[完整目录](docs/project-tree.md)用于定位文件；[开发约定](docs/development.md)统一测试、设备选择和产物管理；[状态](docs/status.md)只保留现役状态。

```bash
JAX_PLATFORMS=cpu pixi run test
pixi run lint
python3 scripts/tools/summarize_final_acceptance.py \
  --selection docs/verification/final-acceptance/selection.json \
  --output tmp/final-acceptance-check.json
```

完整 MPC 数值回归需要[操作手册](docs/runbook.md)中记录的 acados 本地构建。正式训练默认使用 GPU；CPU 用于轻量验证、测试及原生规划器宿主执行。

## 许可与来源

项目使用 [GPL-3.0-only](LICENSE)。各上游代码、移植实现、补丁与外部进程的来源见[第三方声明](THIRD_PARTY_NOTICES.md)及[固定来源清单](third_party/sources.yaml)。公开源代码与研究结果的适用范围，以具体方法配置和评测协议为准。
