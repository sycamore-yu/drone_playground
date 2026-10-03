# ADR 0001：以物理接口组合飞行方法，独立声明求导能力

## Status

Superseded by [ADR 0004](0004-reference-setpoint-and-native-dynamics.md).

## Context

早期宿主需要让学习方法、C++ 规划器和 MPC 共用一条执行链，同时避免把网络内部特征当作跨模块接口。

## Decision

项目最初采用 Traj.、Waypoint、Motion Cmd 作为公共组合边界，并让每个组合独立声明求导边界。PPO、SHAC、BPTT 只表示训练算法，不据算法名称推断网络输出接口。

第一版允许 JAX 仿真与 C++ 规划器共存。SUPER 和 EGO 保留固定 ROS 容器作为原版基线。去容器与可微性分开处理。

## Consequences

- 学习方法和原生规划方法可以通过显式物理数据组合。
- 方法复现与统一条件下的组件比较保持独立身份。
- `Motion Cmd` 作为总括类型后来证明过宽，Reference 与 Setpoint 的语义也需要拆开，因此该接口模型由 ADR 0004 取代。
- 历史交付范围与依据保留在[交付规格](../notes/archive/release-plan.md)。
