# ADR-0009：延迟缓冲状态归属对应环境实例

**Status:** Accepted · 2026-10-09

**Implementation:** 待实现。当前存在传感采样和扫描位姿历史，非零动作送达延迟与测量送达延迟仍不支持。

## Context

无人机收到的命令和方法读到的测量可以晚于产生时刻。历史必须随各并行环境独立推进，支持局部重置及完整训练恢复。仅保存最新动作或最新测量不能表达这个过程。

## Decision

使用延迟缓冲区。命令历史属于对应实例的环境执行状态；测量历史属于现有 `ObservationState`。不将可变队列放入共享的 Controller、Sensor 配置对象或宿主全局变量。

```text
EnvState
  physics
  task
  command buffer state
  observation: ObservationState
    measurement buffer state
```

这个结构表示状态归属，不规定必须新增这些同名类或字段。数组及时间记录属于状态；采集、入队和读取逻辑放在现有物理执行与传感模块。

每个世界只使用当前已经送达的数据。命令未送达时继续执行上一条有效命令；测量未送达时不把新测量提前暴露给 Policy 或 Planner。测量保留实际采集时间、有效掩码和对应采集位姿。

局部 reset 只清理指定世界的历史。完整 Checkpoint 保存缓冲区和相关随机状态；恢复后继续相同时间关系。动作数值经过缓冲区时保留路径导数，延迟索引和调度时间按声明的配置处理。

固定采样步长可使用循环索引；多频率或按秒设置的送达关系可使用时间戳。选择实现以实际时钟需求为准，不新增通用消息队列或调度管理框架。

## Consequences

采样频率、送达延迟、电机响应时间常数分别记录。延迟推进使用仿真时间，不用 `sleep` 代替物理推进。

首次没有已送达测量时的有效性、延迟采样分布、缓冲容量和配置字段仍需在实现设计中确定。数据不足时不能把未来测量当作已送达数据。

零延迟回归应保持现有行为；固定延迟、局部 reset、恢复、独立世界和延迟动作梯度需要通过针对性测试。

## Evidence

- [Isaac Lab DelayedPDActuator](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/actuators/actuator_pd.html)：循环缓冲、物理步延迟和按环境 reset。
- [MuJoCo Playground push_cube](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/franka_emika_panda_robotiq/push_cube.py)：环境状态中的历史动作和观测。
- 当前状态：[EnvState](../../src/drone_playground/simulation/environment.py)、[ObservationState](../../src/drone_playground/simulation/observation.py)、[Measurement](../../src/drone_playground/simulation/sensors.py)。
- 旧仓库 `src/drone_playground/control/delay.py`：动作历史和物理子步送达。它不能作为旧版已支持任意传感送达延迟的证据。
