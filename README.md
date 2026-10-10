# Drone Playground

**v0.2 development branch:** This is the next-generation JAX/Crazyflow research platform.
The frozen v0.1 source is retained on the `v0.1` branch of
[sycamore-yu/drone_playground](https://github.com/sycamore-yu/drone_playground).
The public `main` branch remains on v0.1 until the v0.2 training and evaluation
acceptance criteria are met. v0.2 is not yet a validated release.

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

# 使用 run.json 的 selection.checkpoint 指向的 policy.zip 路径替换下列路径。
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

同一 worker 同时仅供一个客户端使用。原生报告按新 S3（旧 S6）判断：覆盖六个主场景，静态和动态
各至少 10 回合，S01/S02/S03 各至少一次安全到达；动态不设成功率门槛。
逐回合 `episodes.csv`、`trajectories.npz`、`decisions.json` 保留实际控制、求解和失败结果。
原生动态及扩展场景的 `passed: null` 表示诊断报告，没有单场景成功率验收门槛。
学习策略 Navigation 仍按新 L3（旧 C4）每主场景 25 回合、至少 23 次成功判断。

新训练使用 `first_full_validation_pass` 三阶段选模：训练中只在六个导航主场景
各跑 8 回合轻量筛查；通过后按新 L3 的任务门槛做一次完整 checkpoint validation，
首次完整通过的权重停止训练并进行独立冻结 benchmark。正式冻结导航还记录
S06/D06，但它们不参与 L3 主场景门槛；不再要求连续三次完整评测。已启动的旧 run
按各自保存的 `last_of_first_three_consecutive_passes` 协议继续，不能默默改变选模标准。
见 [ADR-0013](docs/adr/0013-three-stage-learning-acceptance.md)。
Checkpoint metadata 的 provenance 保存选择协议、评测索引及累计活动耗时；
恢复后的吞吐率使用本次进程新增交互数除以本次耗时。`training_wall_seconds` 累计各段
活动时间（含初始化、保存与评测，统计至对应记录时刻），不含停机时间；
`session_wall_seconds` / `session_interactions` 描述本次调用。v2 训练归档必须包含
完整训练状态和配置；缺失字段不再推测或补齐。
已通过完整选模的 checkpoint 恢复后直接重跑 benchmark；已有结果保留，新尝试写入
`eval/<编号>/`，实际目录记录在 `run.json` 的 `benchmark_directory`。

## 架构与记录

`Learning → Simulation → Crazyflow`。Learning 管采样、损失、更新与完整状态恢复；
Simulation 管任务、场景、传感器、方法、控制执行、冻结策略评测与回放。
冻结推理从 Simulation 加载 Actor，不依赖 Trainer；训练与推理共用归档校验。
详见 [架构决策](docs/adr/0001-simulation-learning-boundary.md)、
[训练接口](docs/training.md)、[场景资产](assets/scenes/README.md) 和 [回放](docs/replay.md)。

新训练的 `results/<run_id>/` 使用 `config.yaml`、TensorBoard
`events.out.tfevents.*`、Orbax `checkpoints/model_*/`、正式 `episodes.csv`
及按需生成的 `replays/best/`。保留单个 `run.json` 状态索引和
`report.json` 成绩索引，但不再重复生成 `metrics.jsonl` 或多层 `eval/`。
旧 ZIP 训练目录仍按原始布局恢复，不移动正在运行的 checkpoint。
详见 [ADR-0014](docs/adr/0014-results-layout-orbax-tensorboard.md)。

新增控制器、LOTF/PointMass 反向模型、延迟及随机化的命令见
[控制模型使用说明](docs/control-models.md)。`method=controller controller=so3` 选择
SO3；两种 MPC 和 `controller=ideal` 使用同一组合入口。Trajectory 保持宿主实现。

## 当前证据

正式交付只以 [ADR-0015](docs/adr/0015-eight-acceptance-gates.md)
规定的 **S1–S4 / L1–L4 八项验收**汇报。18 单元×3 独立随机种子、
三阶段冻结评测和全部原成功率门槛属于 L3，不再附加历史 C 系列检查表。

### v0.2 从零训练矩阵

本轮正式训练严格使用 `initial_checkpoint=null`：Actor、Critic、优化器均从各
独立种子随机初始化，不加载旧版或参考权重。PPO、APG/BPTT、SHAC 覆盖
Tracking、Racing、Navigation Depth、Navigation LiDAR，共 12 组训练／
18 个静态与动态验收单元，每组三种子。DiffAero 和旧仓库只提供配置经验。

一个管理入口 `tools/train_scratch_matrix.py` 负责 GPU 调度、真实收敛测量与
失败后重新随机初始化试验；所有中间训练、日志、报告和重试参数都存放在
Git 忽略的 `results/scratch_v02/`。`matrix.json` 是唯一的当前训练矩阵汇总，
运行中的任务不得视为成功。失败尝试保留真实记录但不新增阶段性 Markdown。

### 历史 v0.1 证据（不计入当前收敛）

下述收敛和原生飞行成绩属于**旧版归档的历史证据**，不是当前 v0.2 从零训练的结果。
旧 v1 训练状态和冻结权重不会由当前代码加载；保留历史报告和冻结权重的原始文件用于审计。
现行验证与最新实现状态见[验证记录](docs/validation.md)。

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
参考权重迁移。旧版 Depth/LiDAR 随机训练的阶段性记录保留在 Git 历史和
`results/scratch_v02/archive/`，不再作为本轮训练的现行入口。
LiDAR 扩展场景 D06 仍仅为 0–15/25；该场景按规格完整报告，不参与主场景门槛判定。

EGO 与 SUPER 各完成 12 个静态、12 个动态回合，并通过 S6；失败完整保留。
逐场景结果和复现命令见 [原生飞行验收](docs/ros_planner.md)。
Tracking、Racing、Navigation 和动态场景的实际 RScope 原生窗口验证见
[回放验证](docs/replay-validation.md)，包括截图、逐帧核对和正常关闭结果。

工作区的 `results/` 保存实际配置、Checkpoint、报告、轨迹和回放，被 Git 忽略，
不随源码包分发。每次运行的环境和实现身份以其 `run.json` 为准。

## 代码查询

本仓库使用 GitNexus 索引，注册名为 `new_drone_playground`。更新流程与 Wiki 模型
配置见[开发与代码查询](docs/README.md)。

```bash
export PATH="$HOME/.nvm/versions/node/v24.20.0/bin:$PATH"
gitnexus analyze --index-only --name new_drone_playground
gitnexus query -r new_drone_playground 'Trainer update checkpoint'
```

Python 与 C++ 之间的消息合同以 Protobuf 文件为准；索引不自动连接所有跨语言字段引用。
