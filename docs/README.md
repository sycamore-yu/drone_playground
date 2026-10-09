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
| [0005](adr/0005-crazyflow-forward-and-training-derivatives.md) | **Withdrawn**：保留撤回记录；新批准的反向模型选择见 0008 |
| [0006](adr/0006-native-crazyflow-randomization.md) | 使用原生 reset/step pipeline 随机化物理 |
| [0007](adr/0007-run-artifact-layout.md) | 权重与选模成绩同目录；统一独立评测；按需创建轨迹和回放 |
| [0008](adr/0008-independent-backward-dynamics-model.md) | 独立选择 `learning.backward_model`；模型数学实现与训练求导连接分工 |
| [0009](adr/0009-delayed-data-in-episode-state.md) | 命令和测量延迟缓冲归属对应环境实例，支持局部 reset 与恢复 |

2026-10-09 已批准的反向模型选择、任务/动作解耦、控制器注入和 SO3 接入、延迟缓冲、物理随机化及产物布局调整尚未完成代码实现。现有运行目录和 README 命令仍描述当前实现，不能把目标布局直接用于尚未迁移的结果。

## 待审设计与代码审查

[控制接口、模型与方法组合设计](research/control-model-design.md)记录当前职责绑定、完整实现建议、命名比较和 SE3 候选来源。该文档为 `Proposed`；其中的新字段、文件拆分和额外修复不因写入文档而获得批准。

## 运行说明

[训练接口](training.md)、[实验配方](experiments.md)、[原生方法](ros_planner.md)、[回放](replay.md)。
