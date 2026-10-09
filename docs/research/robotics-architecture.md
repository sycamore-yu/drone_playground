# 仿真框架与算法扩展：第一方参考

**研究问题：** 怎样让四旋翼仿真独立运行多种 Planner、Controller 和 Policy，并供学习算法复用？

本文记录外部项目的实现机制，技术取舍在 [ADR-0001](../adr/0001-simulation-learning-boundary.md) 和 [ADR-0002](../adr/0002-crazyflow-upstream-and-style.md) 中确定。

## Crazyflow：物理内核

官方：[仓库](https://github.com/learnsyslab/crazyflow) · [Simulation](https://learnsyslab.github.io/crazyflow/user-guide/sim-overview/) · [Functional API](https://learnsyslab.github.io/crazyflow/user-guide/functional-api/) · [Pipeline](https://learnsyslab.github.io/crazyflow/user-guide/pipelines/) · [MuJoCo/MJX](https://learnsyslab.github.io/crazyflow/user-guide/mujoco/)

| 官方能力 | 使用方式 |
|---|---|
| `Sim`、`SimData`、控制栈与多世界 | 构造四旋翼、发出物理控制量并推进独立环境 |
| `build_step_fn()` / `build_reset_fn()` | JAX 批量步进和可微训练 |
| Step/Reset Pipeline | 必要时插入扰动、随机化和实际控制变换 |
| MJCF/MjSpec/MJX | 装配飞行场景、几何查询、测量与回放 |

Crazyflow 的飞行器动力学由其 JAX 数学实现推进，MJCF/MJX 提供相应几何和接触信息。某些接触记录与实际飞行器刚体响应属于不同功能，需通过运行测试区分。

## Nav2：按角色扩展方法

官方：[Planner 插件](https://docs.nav2.org/jazzy/tutorials/plugin_tutorials/writing_new_planner_plugin/writing_new_planner_plugin/) · [Controller 插件](https://docs.nav2.org/rolling/tutorials/plugin_tutorials/writing_new_controller_plugin/writing_new_controller_plugin/) · [Planner Server](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/core_servers/configuring_planner_server/) · [Behavior Trees](https://docs.nav2.org/rolling/getting_started/nav2_behavior_trees/)

Nav2 通过 `nav2_core::GlobalPlanner`、`nav2_core::Controller` 及 ROS 2 `pluginlib` 提供方法替换；Server 负责算法调度，Behavior Tree 组织规划、控制与失败恢复。

drone_playground 使用其中的**职责接口、配置选择与执行调度分离**：Planner、Controller、Policy 各有明确的物理合同；JAX 方法在进程内执行；原生 ROS/C++ 方法采用相应适配。任务事件仍由 Simulation 定义。

## FlightBench：仿真、学习与原生 Planner

官方：[仓库](https://github.com/thu-uav/FlightBench) · [文档](https://thu-uav.github.io/FlightBench/)

| 原项目路径 | 核心职责 |
|---|---|
| `flightlib/` | Flightmare 仿真与绑定 |
| `flightrender/` | Unity 渲染 |
| `flightros/` | ROS 与仿真控制通信 |
| `flightbench/` | 案例、测试及规划方法基线 |
| `flightrl/onpolicy/` | PPO/MAPPO 和学习环境 |

这种组织将独立运行的飞行算法与学习更新分开。drone_playground 采用更直接的 Crazyflow/JAX 执行路径。

## 机器人仿真与软件实践

| 来源 | 借鉴内容 |
|---|---|
| [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/) 与 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | 物理仿真、任务环境和学习工作流的分工 |
| [MuJoCo](https://mujoco.readthedocs.io/) 与 [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) | 小的物理输入接口、MJCF 资产与任务 reset/step |
| [RotorPy](https://github.com/spencerfolk/rotorpy) | 直接组合车辆、参考和控制器 |
| [Hydra](https://hydra.cc/docs/advanced/instantiate_objects/overview/) | 用配置构造 Python 对象和函数 |
| [Pixi](https://pixi.prefix.dev/latest/python/pyproject_toml/) | 在 `pyproject.toml` 中管理可重建的环境与任务 |
| [RScope](https://github.com/Andrew-Luo1/rscope) | 基于真实 MuJoCo/Brax 状态的回放与查看 |

## 传感器与具名方法来源

| 资料 | 项目参考 |
|---|---|
| [Intel RealSense D435i](https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html) | 深度视场、图像模式与标定 |
| [Livox Mid-360](https://www.livoxtech.com/cn/mid-360/specs) | 扫描方向、刷新率、量程与点云时间 |
| [EGO-Planner](https://github.com/ZJU-FAST-Lab/ego-planner) | 深度感知与局部时标轨迹 |
| [SUPER](https://github.com/hku-mars/SUPER) | LiDAR 地图与时标轨迹 |
| [AllocNet](https://github.com/KumarRobotics/AllocNet) | 轨迹时间分配作为 Planner 内部算法 |
| [MPCC](https://arxiv.org/abs/2108.13205) | 路径进度与控制联合优化 |
| [AC-MPC](https://github.com/uzh-rpg/acmpc_public) | 学习与 MPC 联合方法 |
| [ViTFly](https://github.com/anish-bhattacharya/vitfly) | Depth 策略及速度命令 |

标准 D435i 与 Mid-360 的仿真模型以具体配置说明实际模拟的测量效应，策略网络的缩放与输入处理独立配置。两篇可微导航论文的方法细节见 [学习方法研究](learning-methods.md)。
