# Drone Playground

基于官方 Crazyflow 的四旋翼研究平台：Tracking、Racing、Navigation8，
D435i Depth / Mid-360 LiDAR，PPO、APG/BPTT、SHAC，以及原生 EGO-Planner / SUPER。
目标与验收条件见 [规格](docs/spec.md)，实际执行证据见 [验证记录](docs/validation.md)。

## 安装与运行

Linux x86-64，安装 Pixi 后在仓库根目录执行：

```bash
pixi install --locked
JAX_PLATFORMS=cpu pixi run sim simulation.device=cpu output=results/tracking_sim
pixi run lint
JAX_PLATFORMS=cpu pixi run test
```

`pyproject.toml` / `pixi.lock` 锁定 Python 3.12、官方 Crazyflow revision、JAX/CUDA12、
MuJoCo/MJX 和 RScope。训练配方默认使用 GPU，需要兼容的 NVIDIA 驱动；CPU 诊断同时设置
`JAX_PLATFORMS=cpu` 和 `simulation.device=cpu`。ROS1 worker 使用独立容器，安装与启动见
[原生方法](docs/ros_planner.md)。

所有入口使用 Hydra 配置，`pixi run drone-playground --cfg job --resolve` 可查看解析配置。

```bash
# 三种学习算法：ppo、apg、shac；实验：tracking、racing、navigation_depth、navigation_lidar。
pixi run train experiment=tracking learning=ppo seed=0 output=results/ppo_tracking_s0

# SIGINT/SIGTERM 在当前更新或评测完成后保存；恢复时保持原任务、方法和训练配置。
pixi run train experiment=tracking learning=ppo seed=0 output=results/ppo_tracking_s0 \
  resume=results/ppo_tracking_s0/checkpoints/latest.training.zip

# 使用 selection.json 指向的实际 policy.zip 路径替换下列路径。
pixi run checkpoint_eval experiment=tracking learning=ppo \
  checkpoint=/absolute/path/to/selected.policy.zip output=results/tracking_eval
pixi run benchmark experiment=tracking learning=ppo \
  checkpoint=/absolute/path/to/selected.policy.zip output=results/tracking_benchmark

# 已启动对应 worker 后运行；每主场景 4 回合，共 12 静态 + 12 动态。
pixi run benchmark task=navigation scene=S01 sensor=depth method=ego \
  'benchmark.scenes=[S01,S02,S03,D01,D02,D03]' benchmark.episodes=4 output=results/ego_s6
pixi run benchmark task=navigation scene=S01 sensor=lidar method=super \
  'benchmark.scenes=[S01,S02,S03,D01,D02,D03]' benchmark.episodes=4 output=results/super_s6

# 在桌面会话启动上游原生 viewer；路径来自回合 report.json 的 replays 字段。
pixi run replay replay_path=/absolute/path/to/episode.mj_unroll
```

同一 worker 同时仅供一个客户端使用。原生报告按 S6 判断：覆盖六个主场景，静态和动态
各至少 10 回合，S01/S02/S03 各至少一次安全到达；动态不设成功率门槛。
逐回合 `episodes.csv`、`trajectories.npz`、`decisions.json` 保留实际控制、求解和失败结果。
原生动态及扩展场景的 `passed: null` 表示诊断报告，没有单场景成功率验收门槛。
学习策略 Navigation 仍按 C4 的每主场景 25 回合、至少 23 次成功判断。

训练连续三次 checkpoint evaluation 达标后，按唯一支持的
`last_of_first_three_consecutive_passes` 规则选择第三次权重，再运行独立 benchmark。
Checkpoint metadata 的 provenance 保存连续通过数、评测索引及累计活动耗时；
恢复后的吞吐率使用本次进程新增交互数除以本次耗时。`training_wall_seconds` 累计各段
活动时间（含初始化、保存与评测，统计至对应记录时刻），不含停机时间；
`session_wall_seconds` / `session_interactions` 描述本次调用。旧归档缺少的计数从零开始，
未知历史耗时保持 `null`。历史进程写出的记录不会被本次代码修改补算。
已完成 C5 的 checkpoint 恢复后直接重跑 benchmark；已有结果保留，新尝试写入
`benchmark-*`，实际目录记录在 `run.json` 的 `benchmark_directory`。

## 架构与记录

`Learning → Simulation → Crazyflow`。Learning 管采样、损失、更新与完整状态恢复；
Simulation 管任务、场景、传感器、方法、控制执行、冻结策略评测与回放。
冻结推理从 Simulation 加载 Actor，不依赖 Trainer；训练与推理共用归档校验。
详见 [架构决策](docs/adr/0001-simulation-learning-boundary.md)、
[训练接口](docs/training.md)、[场景资产](assets/scenes/README.md) 和 [回放](docs/replay.md)。

每次运行写入 `results/<run_id>/`：`config.yaml`、`run.json`、`events/metrics.jsonl`、
`checkpoints/`，以及评测目录内逐场景报告、回合 CSV、轨迹和可选 RScope 记录。
评测保留所有失败分母；checkpoint evaluation 与 benchmark 使用分离的种子区间。

## 当前证据

可复核的代码检查在 [CLI 测试](tests/test_cli.py)、[学习测试](tests/test_learning.py)、
[回放测试](tests/test_replay.py) 与 [ROS Planner 客户端测试](tests/test_ros_planner.py)。
原生求解器实测见 [integration-results.json](src/ros_planner_worker/integration-results.json)，
它验证求解接口，不等同于完整飞行验收。

Tracking、Racing、Depth/LiDAR Navigation 的三算法、三种子正式运行全部完成，
共 36 次训练、5,400 个冻结评测回合，**18 个验收单元全部通过**。
逐种子证据见 [验收表](results/acceptance/current.md) 与 [验收说明](docs/acceptance.md)。
Racing 和 Navigation 使用共同初始化的配方单独记录预训练来源及成本，
详见 [实验调整](docs/experiments.md) 和 [吞吐验证](docs/performance.md)。
Racing 的共同初始化由本项目 APG 从随机权重训练得到；Depth/LiDAR 的上述配方包含历史
参考权重迁移。另已启动三算法、三种子的 [Depth/LiDAR 从零训练](docs/scratch-training.md)，
其收敛状态单独报告。
LiDAR 扩展场景 D06 仍仅为 0–15/25；该场景按规格完整报告，不参与主场景门槛判定。

EGO 与 SUPER 各完成 12 个静态、12 个动态回合，并通过 S6；失败完整保留。
逐场景结果和复现命令见 [原生飞行验收](docs/ros_planner.md)。
Tracking、Racing、Navigation 和动态场景的实际 RScope 原生窗口验证见
[回放验证](docs/replay-validation.md)，包括截图、逐帧核对和正常关闭结果。

工作区的 `results/` 保存实际配置、Checkpoint、报告、轨迹和回放，被 Git 忽略，
不随源码包分发。每次运行的环境和实现身份以其 `run.json` 为准。

## 代码查询

本仓库已建立 GitNexus 索引，注册名为 `new_drone_playground`。代码更新后可刷新：

```bash
gitnexus analyze --index-only --name new_drone_playground
gitnexus query -r new_drone_playground 'Trainer update checkpoint'
gitnexus query -r new_drone_playground 'RosPlanner rollout_ros'
```

Python 与 C++ 之间的消息合同以 Protobuf 文件为准；索引不自动连接所有跨语言字段引用。
