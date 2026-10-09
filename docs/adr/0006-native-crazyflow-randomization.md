# ADR-0006：通过 Crazyflow 原生流水线随机化物理

**Status:** Accepted · 2026-10-09

**Implementation:** 决策已批准；物理参数随机化和运行中扰动尚待实现。现有初态采样继续保留。

## Context

平台需要在相同任务下研究物理参数误差和外部扰动。Crazyflow 已提供 `SimData.params`、reset/step pipeline 及纯函数构建入口。另建一层环境包装或修改仅用于场景的 MJX 参数，会增加状态管理成本，或无法改变真正的飞行物理。

## Decision

物理参数随机化在官方 `reset_pipeline` 中完成，并通过 `build_reset_fn()` 构建实际 reset 函数。参数写入负责飞行推进的 `SimData.params`。每个并行世界使用独立随机状态；只更新 reset mask 选中的世界。

每次从标称参数采样，不能在上一回合的随机参数上累计乘系数。同步维护相关派生量，例如惯量及逆矩阵。所选模型没有使用的参数不提供无效随机化选项。

外力、外部力矩和阵风在官方 `step_pipeline` 的载荷/积分调用位置接入，遵循原生输入的坐标和单位。随机过程按仿真物理时间推进，不按宿主循环次数或墙钟时间推进。流水线完成装配后再构建 `build_step_fn()`。

实现放在 Simulation 的小型随机化模块中，不恢复旧版 Brax `EnvironmentEffects` 包装体系。函数、随机状态和参数分布使用现有配置与状态设施，不新增事件管理框架。

控制器标称参数和仿真真实参数分开。是否向方法提供随机后的真实参数属于实验的信息权限，不能在随机化时自动改变。

## Consequences

保留原有标称验收配方。鲁棒性实验使用明确的分布和冻结评测种子，并保存实际物理参数及随机状态，使 Checkpoint 恢复继续同一实验。

MuJoCo Playground 的 `domain_randomize(model, rng)` 提供纯函数及批量参数的参考。当前飞行方程由 Crazyflow 执行，因此直接使用其参数与 reset/step pipeline，不并行维护第二份 MJX 飞行参数。

## Evidence

- [ADR-0002](0002-crazyflow-upstream-and-style.md)：优先使用官方 Crazyflow 公共 API。
- [Crazyflow 原生 API](https://learnsyslab.github.io/crazyflow/user-guide/oo-api/) 与 [Pipelines](https://learnsyslab.github.io/crazyflow/user-guide/pipelines/)。
- 当前锁定依赖 `crazyflow/sim/sim.py`：`reset_pipeline`、`step_pipeline`、`build_reset_fn`、`build_step_fn`，于 2026-10-09 核对。
