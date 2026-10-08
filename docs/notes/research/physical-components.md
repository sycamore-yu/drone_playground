# 物理组件组合的工程边界

现役公共物理边界分为 Reference、Setpoint 和 Actuation：

- `Waypoint` / `Trajectory`：规划或任务参考；
- `StateSetpoint` / `AttitudeSetpoint` / `RateSetpoint`：控制设定值；
- `ForceTorque` / `MotorRPM`：原生执行器输入。

这些类型分别位于 `references.py` 和 `control/setpoints.py`。不存在通用
`MotionCommand` 或按向量宽度解释语义的 `COMMANDS` 表。

## 神经几何输出

`PhysicalActionDecoder` 给冻结策略声明 Waypoint / Trajectory 的米制尺度、锚点、
时域和输出维度。训练中的几何输出由
`control/controllers/trajectory_jax.py` 转换为实际姿态/推力，再进入同一延迟、
Controller 和 Dynamics 链。

现役入口是 `experiment=control/geometric`。它复用 PPO/BPTT/SHAC 等训练算法，
不建立新的算法身份。检查点保存物理解码、观测规格和目标来源；相同输出宽度不能
替代这些语义检查。

## Pipeline 与外部方法

`runtime/pipeline.py` 只负责 stage 构造、频率、缓存、有效期和重置。阶段由
`_target_` 直接实例化，例如：

```yaml
method:
  implementation: pipeline
  output: trajectory
  stages:
  - _target_: drone_playground.learning.inference.FrozenNeuralCommand
    checkpoint: /path/to/policy.pkl
    frequency_hz: 10
  - _target_: drone_playground.planning.minimum_jerk.MinimumJerkPlanning
    frequency_hz: 10
```

原生 C++ 服务使用
`drone_playground.integrations.service.NativeServicePlanner`，底层 transport
由 `drone_playground.integrations.rpc` 提供。SUPER/EGO 额外需要 ROS1 deployment adapter；
其容器、补丁和构建脚本位于 `docker/ros1/`。

## MPC

AttitudeMPC 与 SamplingMPC 位于 `control/controllers/mpc/`。它们是真实
Controller/trajectory-tracking 实现，不通过配置标签冒充 Planner。外部 Trajectory
必须覆盖 MPC 所需未来时域；不足时显式拒绝或返回无计划，不外推未来参考。

## 求导边界

JAX Policy、Reference、可微 Controller 和支持的 Dynamics 可按实际实现传播梯度。
RPC、C++、ROS 和 acados 不因为接入同一接口而自动可微。含这些阶段的组合用于宿主
执行和质量评测，不声明端到端 BPTT/SHAC 梯度。

当前重构和短闭环证据集中在
`artifacts/verification/direct-composition-v4/`；历史实验凭据仍保留原文件，
不因接口迁移改写。
