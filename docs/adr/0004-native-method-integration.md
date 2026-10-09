# ADR-0004：原生 C++／ROS 方法接入

**Status:** Accepted · 2026-10-08

## Context

EGO-Planner 和 SUPER 的原生求解器使用 C++／ROS，平台物理执行使用 Crazyflow/JAX。两端需要稳定的时间、测量与轨迹合同。

## Decision

- **Transport**：Protobuf/gRPC。由 EGO-Planner 和 SUPER 的实际消息需求定义服务和数据合同，覆盖初始化、回合重置、决策与关闭。正式接入时创建最小 `.proto`。
- **Runtime**：C++ 服务实现生成的 gRPC 接口。EGO/SUPER 在 ROS1 Noetic 容器中运行，由 worker 转换原生消息。
- **Data**：传递带单位、坐标、时间与有效期的状态、Depth/LiDAR、目标及路径／轨迹／控制输出。
- **Execution**：Simulation 消费原生决策，由对应控制器生成物理命令并推进 Crazyflow。

## Consequences

方法可以独立部署并复用统一 Task、Benchmark 和 RScope 回放。RPC 处理通信与进程隔离，JAX 方法直接进行进程内执行；ROS2 算法按实际作者依赖另行提供消息适配。

2026-10-09 的命名与布局：Python 的 `RosPlanner` client 与生成的 Protobuf/gRPC
代码集中在 `src/drone_playground/simulation/ros_planner/`；独立 C++ 工程位于
`src/ros_planner_worker/`，与 Python 包并列。手写 client 与生成代码分文件保存，
保证重新生成不会覆盖手写实现。RPC 使用 `drone_playground.ros_planner.v1.RosPlanner`；
更名前的 worker 镜像需要重建。
