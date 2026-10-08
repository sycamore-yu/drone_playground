# ADR 0002：以真实飞行组件直接装配 DroneEnvironment

## Status

Accepted.

## Context

现有环境构造把方法名、任务适配器、控制器、训练器和评测入口的兼容规则集中在 `composition.py` 与 `environment.py`。中央代码因此需要了解 PPO、BPTT、SHAC、原生规划器、MPC 和不同导航实现。

项目需要继续支持学习控制、轨迹规划、MPC、Crazyflow、LOTF 和 PointMass。增加新方法不应继续增加中央 `if/elif` 分派。

## Decision

核心环境按六个真实组件直接装配：

```text
DroneEnvironment(
    dynamics=...,
    controller=...,
    reference=...,
    scene=...,
    sensor=...,
    task=...,
)
```

各组件职责如下：

- `dynamics` 负责真实前向物理推进。
- `controller` 把上游 Reference 或 Setpoint 降到动力学可执行的控制层级。
- `reference` 提供环境内生的目标运动，例如悬停、Figure-8 或赛道参考。
- `scene` 提供几何、运动障碍和实例事实。
- `sensor` 从真实物理状态与场景生成测量。
- `task` 负责成功、失败、进度、观测语义、奖励和终止。

Planner、Policy 和 MPC 不成为 Environment 的特殊分支。Planner 产生 Reference；Controller 或 Policy 产生 Setpoint。MPC 按其真实职责归入 Controller，而不是形成独立的环境类别。

配置校验位于 `configuration.py`，运行模式分派位于 `app.py`；不保留单独的 `composition.py`。中央入口不维护具体方法、tracker、adapter、trainer 或 evaluator 的兼容矩阵。

不新建全局 `EnvironmentState`。环境边界继续使用 Brax `State`。`pipeline_state` 保存环境内部运行状态，并允许 Crazyflow、LOTF 和 PointMass 保留各自原生物理状态。

## Consequences

- 新方法通过实现 Planner、Policy 或 Controller 边界接入，不要求修改中央环境分派。
- Task 不再选择 trainer 或 evaluator。
- 现有 `task.adapter` 只要能由真实组件组合表达，就应删除。
- Tracking、Racing、Navigation 的语义应靠近各自 Task，而不是分散到全局 observer、objective 和 builder 中。
- 迁移必须按任务逐步进行，并用同种子短 rollout 验证物理结果没有非预期变化。
- 该决策允许源码目录继续按职责划分，但禁止为了“可组合性”新增 Manager 或万能组件框架。
