# ADR-0008：独立选择反向求导的动力学模型

**Status:** Accepted · 2026-10-09

**Implementation:** 待实现。当前仍使用官方 Crazyflow 前向和已有的时间梯度衰减，尚无 `learning.backward_model`。

## Context

用户需要保持现有前向模型选择，同时使用 LOTF 或 PointMass 的动力学方程计算策略更新所需的梯度。反向模型具有状态、控制输入、模型参数和状态转移方程；梯度衰减不能代替它。

ADR-0005 曾在未经批准时写入模型范围限制，已撤回。本文件记录本轮明确批准的选择与职责，不恢复该限制。

## Decision

保留 `simulation.dynamics` 的现有含义和官方 Crazyflow 前向选择。增加独立的 `learning.backward_model`：未指定时，对实际前向模型求导；选择 LOTF 或 PointMass 时，用对应模型计算训练所需的导数。

```yaml
simulation:
  dynamics: first_principles

learning:
  backward_model: point_mass_lag
```

上面的字段是已批准的目标配置，当前程序尚不接受它。省略反向模型不创建第二个模型实例。已有梯度衰减作为显式训练选项单独记录，不隐含选择反向动力学。

职责如下：

| 位置 | 职责 |
|---|---|
| `simulation/dynamics/` | LOTF、PointMass 等动力学数学实现，可独立调用 |
| `learning/` 的训练转移调用处 | 选择前向与求导模型，连接自动微分 |
| `simulation/methods.py` 及具体控制器模块 | Planner、Controller、Policy 的组合与执行 |
| `Environment` | 物理时钟、任务、传感和实际状态推进 |

PPO、APG、SHAC 共用环境与训练转移入口。PPO 不通过动力学求策略梯度，不因配置了反向模型而额外运行它。需要物理路径导数的算法共用选定的求导实现。

同一数学模型只维护一份。模型可独立用于求导、预测或另行批准的仿真实验。此次接入不要求迁移 LOTF 的 ROS 部署、Betaflight 运行状态或旧版环境。

## Consequences

在相同参数、初态、命令和随机状态下，仅更换反向模型应保持前向状态、事件和测量相同。训练更新后的策略可以不同。

反向模型输入必须与求导边界的物理输入对齐。实现前明确状态映射、单位、坐标、控制转换、步长，以及未建模变量的导数。不能把姿态指令直接解释成角速度，也不能用一个梯度缩放因子冒充 PointMass。

模型参数和作用范围随训练配置、Checkpoint 保存。完整恢复使用相同求导配置；冻结推理不加载只用于训练的反向模型。

本决定没有禁止未来新增前向模型，也没有批准任意前向/反向组合。具体变量映射和模块命名见[待审设计](../research/control-model-design.md)。

## Evidence

- [LOTF 原论文 III-B、III-E](https://arxiv.org/html/2508.21065v2)：解析动力学及解析模型求导。
- [LOTF 官方源码](https://github.com/uzh-rpg/learning_on_the_fly/blob/master/lotf/objects/quadrotor_obj.py)：`_step_jvp` 调用 `jax.jvp(simplified_dyn, ...)`。
- [DiffAero 原论文 III-A、IV-D](https://arxiv.org/html/2509.10247v1)：多动力学与加速度策略的部署转换。
- [JAX 自定义导数](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)。
- 当前入口：[Environment](../../src/drone_playground/simulation/environment.py)、[Trainer](../../src/drone_playground/learning/trainer.py)、[梯度衰减](../../src/drone_playground/learning/losses.py)。
