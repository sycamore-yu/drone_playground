# 文档入口

[功能规格](spec.md)定义研究范围；[验证记录](validation.md)和[验收说明](acceptance.md)描述已有执行证据。术语使用根目录的 [GLOSSARY.md](../GLOSSARY.md)。

## 架构决策

`Accepted` 表示用户已明确批准设计，提问或要求评估不构成批准。`Withdrawn` 表示提案已撤回，不作为实施依据。每份 ADR 的实现说明区分当前能力和待实现改动，不能用 ADR 代替运行证据。

| 决策 | 内容 |
|---|---|
| [0001](adr/0001-simulation-learning-boundary.md) | Simulation/Learning 分工；2026-10-09 补充构造阶段的方法组合和两种执行方式 |
| [0002](adr/0002-crazyflow-upstream-and-style.md) | 官方 Crazyflow、依赖及代码规范 |
| [0003](adr/0003-project-scope.md) | 首发研究范围和验收 |
| [0004](adr/0004-native-method-integration.md) | 原生 C++/ROS 接入 |
| [0006](adr/0006-native-crazyflow-randomization.md) | 使用原生 reset/step pipeline 随机化物理 |
| [0007](adr/0007-run-artifact-layout.md) | 权重与选模成绩同目录；统一独立评测；按需创建轨迹和回放 |
| [0008](adr/0008-independent-backward-dynamics-model.md) | 独立选择 `learning.backward_model`；模型数学实现与训练求导连接分工 |
| [0009](adr/0009-delayed-data-in-episode-state.md) | 命令和测量延迟缓冲归属对应环境实例，支持局部 reset 与恢复 |
| [0010](adr/0010-training-only-privileged-ppo-critic.md) | 训练专用特权 PPO Critic；Actor 与冻结评测使用原设备观测 |
| [0011](adr/0011-independent-ppo-goal-observation.md) | 独立 PPO 对照增加目标高度与距离；原验收批次保留十维观测 |

2026-10-09 的控制模型改动已整合到 `main`：反向模型、任务/动作解耦、控制器注入、SO3、两种 MPC、理想跟踪、延迟缓冲、随机化和新产物布局已接入。原有训练结果未迁移，旧训练作业已暂停。用法见[控制模型说明](control-models.md)，原分支验证及合并记录见[实施记录](plans/control-models-20261009.md)。

## 设计与代码审查

[控制接口、模型与方法组合设计](research/control-model-design.md)记录已批准的改进及原代码审查。用户取消了 Trajectory 的 JAX 化要求，SE3 继续暂缓；没有将这两项扩展加入实现。

## 运行说明

[训练接口](training.md)、[实验配方](experiments.md)、[原生方法](ros_planner.md)、[回放](replay.md)。
