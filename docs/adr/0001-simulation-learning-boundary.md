# ADR-0001：Simulation 与 Learning 的职责划分

**Status:** Accepted · 2026-10-08

## Context

平台需要让优化控制、原生 Planner 和冻结学习策略共享任务、物理与评测，同时支持 JAX 的批量及可微训练。

## Decision

使用一个可安装 Python 包，内部划分 **Simulation** 与 **Learning**。

- **Simulation**：Task、Scene、Sensor、运行时 Planner/Controller/Policy、rollout、benchmark、RScope 回放。
- **Learning**：PPO、APG/BPTT、SHAC 的采样、损失、优化器、参数更新与训练状态。
- **Dependency**：Learning → Simulation → Crazyflow；冻结方法由 Simulation 独立运行。
- **Extension**：采用 Nav2 按 Planner/Controller 职责替换、配置选择具体方法、调度与算法分离的原则。支持 Planner→Controller 和直接 Policy／联合 MPC 两类运行链。

## Consequences

方法以物理数据、信息权限和时间合同接入。Simulation 直接运行 JAX 方法；原生 C++/ROS 由独立进程适配。训练与前向共用同一套任务判定及 Crazyflow 状态转移。

2026-10-09 的职责细化：Simulation 的 `evaluation` module 选择任务、Method 对应的
评价协议与种子分区，并在记录前确定 C2/C3/C4 或原生 S6/diagnostic 判定。
记录实现保存传入的判定和实际 Episode 数据；CLI 调用该 module。
独立验收收集器继续从保存的数据核验结果，不依赖运行时评测判定。

## 2026-10-09 方法组合补充

**Status:** Accepted。**Implementation:** 待实现控制器注入与构造阶段的方法组合；当前两种执行路径已存在。

Hydra 在构造阶段创建 Planner、Controller 或 Policy。需要规划跟踪时，使用 `PlannerController(planner, controller)`；直接输出控制目标的 Policy 或联合方法直接执行。运行循环调用已经构造的方法，不再根据 `ego`、`super`、`policy` 等算法名字分派。

`PlannerController` 只负责规划周期、有效轨迹缓存、控制周期及两者状态。路径搜索、地图、优化器和网络仍属于具体方法。它是本项目这条实际组合的实现，不扩展为任意阶段的通用 Pipeline。

Environment 保留物理时钟、命令执行、任务判定和传感采样。轨迹跟踪控制器从外部注入，不在 Environment 内固定创建 Mellinger。运行方法与执行过程共享同一控制输出合同。

保留 JAX 和宿主两种执行方式。JAX 路径使用纯函数、显式参数/状态/RNG 及 `jit`；宿主路径调用 C++/ROS 或宿主求解器。二者共用 Environment、任务规则、评测和产物格式。普通 RPC 调用不进入路径式自动微分。

当前实现位置为 `simulation/runner.py` 的 `rollout()` / `_rollout_kernel()` 和 `rollout_ros()`，不是两个已经实现的 Runner 类。后续沿用两种执行方式，不以新增类或拆分文件作为完成标准。

## 2026-10-09 控制接口与独立数学实现补充

**Status:** Accepted。**Implementation:** 待实现任务与动作接口解耦、SO3Controller 接入；现有模型、网络及执行路径仍有固定配方绑定。

Task 不绑定动作层级。Tracking、Racing 和 Navigation 定义参考需求、奖励/指标及终止规则；动作维数、单位、坐标和范围由所选控制接口决定。Actor 输出遵守该接口，不根据任务名或传感器名决定动作含义。

模型和控制器的数学实现可以在训练 CLI 之外独立调用。一个动力学模型可用于求导、预测或相应仿真；一个控制器可用于冻结评测或具备所需导数的训练。学习算法面对统一环境与转移接口，增加模型或控制器不复制 PPO、APG、SHAC 的采样逻辑。

保留控制器注入。批准加入基于 EGO-Planner 原实现的 `SO3Controller`，与现有 `MellingerController` 处于同一轨迹跟踪选择层级，由相同的 PlannerController 组合使用。它不新增一个架构层级。

SO3 的输入、输出与 JAX 移植需要对照[原 SO3Control](https://github.com/ZJU-FAST-Lab/ego-planner/blob/bfda51284c8c1b476043255a8145ef925a3778a5/src/uav_simulator/so3_control/src/SO3Control.cpp)。接口转换、名义机体参数及剩余低层控制须记录。加入 SO3 不要求复制 EGO 的整个仿真器。

模块是否拆分或改名、完整配置布局、SE3Controller 是否加入仍是[待审设计](../research/control-model-design.md)中的建议。此补充只记录上述明确批准的内容。
