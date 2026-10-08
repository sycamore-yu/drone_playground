# 可组合无人机环境

公开入口使用 Hydra 组合和实例化。项目不建立 Environment、Dynamics、Sensor、Controller 或 Algorithm registry。`drone_playground.load("hovering")` 只加载环境，不选择 PPO 或其他训练配方。

## 配置与构造

`configuration.py` 负责 Hydra 组合、校验和冻结检查点的显式覆盖；`app.py` 负责执行分派。CLI 与 Python 的 `run_experiment(..., overrides=...)` 使用同一个配置准备过程。`composition.py` 已删除。

`environments/factory.py` 解析运行条件并创建六个组件。`environments/base.py` 的 `DroneEnvironment` 接收组件对象；`environments/initialization.py` 建立并持有公共仿真、控制器绑定和时钟。Task 可以返回任务特有的原生仿真资源，但不再在 `bind()` 中补齐公共执行基础设施。初始化失败时释放已创建的资源。

| 配置 | 职责 | 实现位置 |
|---|---|---|
| `env.dynamics` | 真实前向动力学、原生积分和物理参数 | `dynamics/` |
| `env.controller` | 将输入降低到动力学支持的物理控制量 | `control/controllers/` |
| `env.reference` | 悬停、figure-8、随机样条或赛道参考 | `references.py` |
| `env.scene` | 场景资产、实例、几何和运动参数 | `environments/scenes/` |
| `env.sensor` | 当前场景与机体状态产生的测量 | `environments/sensors/` |
| `env.task` | 初始化、观测、事件、奖励或损失 | `environments/tasks/` |
| `method` | 运行时策略、规划器、控制器或已配置模块链 | 对应的学习、规划、控制或外部接入模块 |
| `algorithm` | 参数更新、优化器、展开时域和导数规则 | `learning/algorithms/` |
| `network` | 编码器、策略、价值网络和记忆结构 | `networks/` |
| `training` | 种子、预算、训练分布、选模和恢复 | `learning/` |
| `evaluation` | 冻结执行、评测入口、cases 和记录开关 | `evaluation/` |
| `runtime` | 设备、宿主/JAX、传输延迟、输出目录 | `runtime/`、`app.py` |

`env.task.observation` 声明输入字段和编码。`env.task.reward` 或 `env.task.loss` 声明任务评价。任务配置没有 trainer、adapter 或 evaluation entrypoint；训练入口是 `algorithm.trainer`，评测入口是 `evaluation.entrypoint`。

`experiment=...` 是实验配方。它组合已有组件，不是另一份构造器注册表。`learning/brax_configuration.py` 只在 Brax 接口边界展开已解析的训练参数；重名参数会报错。它不维护另一组实验默认值。

## 状态与推进

环境外层继续使用 Brax `State`。`pipeline_state` 保留任务需要的原生物理容器；`obs` 是已声明的观测；`info` 保存回合状态。项目不建立第二个通用 DroneState 或 EnvironmentState。

统一物理入口为：

```python
next_state = dynamics.step(state, control, dt)
```

`control` 是执行控制器输出的物理类型。`dt` 是秒。Crazyflow 调用自身公共 `build_step_fn()` 生成的推进函数；LOTF 使用其原生内环和积分器；PointMassLag 使用原有加速度与滞后方程。接口统一，不替换各后端的数值实现。

`env.freq` 是环境控制频率。`env.physics_freq` 指定无原生时钟模型的物理频率，或校验实际后端频率。子步数由环境统一计算，Task 不维护另一份时钟配置。物理和策略更新频率不同；子步循环不改变原生控制器的更新频率。

`control/transition.py` 的 `ActionTransition` 共用一个 JAX 子步循环，执行动作到达、物理推进、事件累计和可选的首次终止冻结。Task 返回事件规则，不自行实现物理循环。Navigation 的碰撞在每个物理步检查，周期内保留是否碰撞和最小间距；普通 Crazyflow Navigation 仍在控制周期末返回，PointMass 仍冻结在首次终止子步。Tracking/Racing 没有中途检查需求的路径可批量调用后端；Racing 保留 LSY 的原生周期末接触与过门判定。

原论文 PointMass 的控制周期离散映射与连续子步递推数值不同。其专用 transfer evaluator 使用同一个执行器的 `sample_from_start` 参数，在物理检测时刻从周期初态计算原映射；无事件时终态与原训练映射一致。适配后的延迟导航和控制任务使用逐物理步递推。训练中的连续损失路径仍保留原有采样频率，不借重构改动论文方程或梯度。

LOTF high-fidelity 与 simplified 是动力学选择，不拥有独立训练算法、Task 或运行角色。`algorithm.gradient.transition` 明确选择 direct、exponential 或后端支持的 `simplified_dynamics_jacobian`。未支持的导数组合会由实际组件拒绝。

## Reference、Setpoint 与 Actuation

| 类型 | 含义和单位 |
|---|---|
| `Waypoint` | 世界系有序位置，米；容差和有效期明确 |
| `Trajectory` | 分段时间多项式，世界系 xyz/yaw；提供位置、速度和加速度 |
| `StateSetpoint` | 显式位置、速度、净加速度、yaw 或 yaw-rate 字段；缺失字段不参与控制 |
| `AttitudeSetpoint` | 世界系 roll/pitch/yaw，弧度；总推力，牛顿 |
| `RateSetpoint` | 总推力，牛顿；机体系角速度，弧度/秒 |
| `ForceTorque` | 总推力，牛顿；机体系力矩，牛顿米 |
| `MotorRPM` | 原生电机顺序的四路转速 |

`references.py` 和 `control/setpoints.py` 定义这些边界。没有通用的扁平 MotionCommand 或 COMMANDS 表。控制器读取类型及字段，不能通过向量宽度推断物理含义。

姿态输入跳过位置控制。角速度输入跳过位置和姿态控制。Crazyflow 的原生角速度数组顺序为 `[wx, wy, wz, T]`；LOTF 为 `[T, wx, wy, wz]`。转换只在后端边界发生。Crazyflow 的角速度和直接执行器输入要求 first-principles 模型。

主机和 JAX 使用同一 Waypoint/Trajectory 数学表示。主机检查非法时间和非有限值；JAX 保留批处理和梯度。轨迹不得在有效区间外外推。已声明的几何策略头通过 `PhysicalActionDecoder` 和 JAX 轨迹跟踪器进入延迟与物理链。任意 RPC/C++ 模块链不被声明为端到端可微。

## 方法状态与运行调度

PPO、BPTT、SHAC 和 DVA 是训练算法。训练得到的 Policy 是运行方法。策略记忆属于策略实例；MPC 的 warm start 和求解器缓存属于控制器；延迟队列、传感器历史和重置键属于环境实例。

`runtime/pipeline.py` 只构造已配置模块并处理调用频率、缓存、有效期和重置。目标航点在 `planning/goal.py`，最小 jerk 轨迹在 `planning/minimum_jerk.py`，冻结策略在 `learning/inference.py`，原生服务在 `integrations/`。MPC 数值实现保留在 `control/controllers/mpc/`。

外部运行代码统一归入 `integrations/`。`integrations/ros1/` 管理 EGO/SUPER 的生命周期和 ROS 消息转换；`integrations/rpc/` 是内部通信实现；`integrations/service.py` 保留任意语言服务的通用接入类 `NativeServicePlanner`。Hydra 直接实例化具体方法，不经过 `create_native_planner` 和 `build_*` 转发。

`docker/ros1/` 只提供 ROS Noetic、C++ 依赖和原算法的可重复部署环境。`tests/native/interop_server.cc` 直接实现 gRPC 生成接口，是协议测试服务，不是规划器或公开 SDK。根目录不再存在 `native/`，包根目录不再存在 `rpc/`。

方法、Pipeline、Controller 和回放归档之间传递同一个 `runtime.Decision`。RPC 仅在进出进程时编解码。轨迹采样由 Controller 完成一次；上游真实控制样本单独保留。检查点与显式运行覆盖在 `configuration.py` 合并，评测器使用已解析配置。进程内策略不经过 RPC；冻结跟踪策略的参考时域不足时返回 `Decision(status="no_plan")`。

低频输出只能在其原有效期内复用。缓存命中不刷新生成时间。无解或过期结果不再传给下游。重置清空模块缓存、调用计数、循环记忆和求解器历史。一次失败不会重置批次中仍在运行的其他实例。

## 观测与训练

观测组件声明字段、形状、单位和坐标。已有状态与感知网络继续使用原有向量布局，保留权重树；循环加速度环境的公共观测使用 `state`、`sensor`、`valid` 字典。自动重置按 pytree 操作，不假定观测一定是一个数组。

检查点保存解析配置、输入规格、动作维度、物理解码合同和参数摘要。恢复不能把相同宽度的另一组输入解释为原输入。完整恢复还保存优化器、随机键、计数及必要环境状态。PPO 参数热启动与完整优化器恢复分别记录。

`artifacts/checkpoints.py` 只保存和校验数组树及元数据；`artifacts/training_state.py` 共用 RNG 与优化器状态序列化。`learning/inference.py` 重建网络及冻结策略，`learning/checkpointing.py` 负责训练快照、选优和公共循环训练编排。各算法保留自己的 rollout、损失、参数更新和恢复限制。

`MotionNavigationReward` 位于 `environments/tasks/rewards.py`，由 Task 写入 `State.reward`；轨迹损失及辅助预测损失留在学习侧。共享的精确范数及零点 JVP、未归一化的 xyzw 旋转公式和单位 pseudo-Huber 在 `numerics.py`。相机光学外参仍属于传感器，碰撞球偏移仍属于场景几何。

训练创建 `train` 环境，训练内选模和最终冻结评测创建 `eval` 环境。运行角色只有这两个。训练分布、初态随机化、测量噪声、动作误差和动力学随机化在构造前解析；名义评测不继承训练扰动。

`termination` 与 `truncation` 保留不同语义。有 critic 的算法用重置前末态处理人工截断；真实终止不继续 bootstrap。奖励公式、参数及首次事件顺序不因架构迁移改变。

## 场景资产与回放

固定几何的物理源位于 `assets/scenes/` 的 MJCF。Navigation8 仍是 S01/S02/S03/S06、D01/D02/D03/D06；`catalog.xml` 组织场景和非几何元数据，独立场景 XML 保存实际图元，`boundary.xml` 保存共同边界。运动参数存于模型数据，当前位置按仿真时钟计算并写入运行态。

传感、净空和碰撞查询使用从编译模型提取的 `SceneBank` 数组。固定场景回放直接附加同一 MJCF；程序生成的训练场景由场景模块序列化其已采样参数。可视化模块不另造一份障碍几何。

LSY 赛道的位置、形状、门顺序和初态来自 `lsy_level0.xml`。原生任务代码继续处理过门和接触规则。非几何上游参数由 YAML 明确提供。Crazyflie 2.x 回放模型也作为独立 MJCF 资源安装。

场景直接加载 MJCF，并检查场景编号、图元、静态/动态标记和实际参数。环境构造不要求预先登记资产 SHA，也不读取离线验收报告。历史几何审查归档在 `artifacts/verification/navigation-mjcf/verification.json`；拓扑检查只由离线验证工具显式运行。Benchmark 固定场景名称、初态、时钟和指标，记录实际运行配置。

离线拓扑和目录审计在 `environments/scenes/catalog_validation.py`，不进入场景加载链。`visualization/navigation_replay.py` 统一导航轨迹导出，保留真实动作前位置、速度、初始帧和终止帧。`rscope_io.py` 打包模型与写出回放；`rscope_publish.py` 将已完成的回放提供给查看器。两者使用 `_rscope_files.py` 中同一套锁及原子写入，发布不改变源运行，追加不替换被监听目录。

## 外部接入与检查数据

Python/C++ 共用 `integrations/rpc/proto/algorithm.proto` 的 v2 协议。初始化、重置、一步调用、输出有效期和关闭均明确。EGO-Planner 与 SUPER 保持原 ROS 节点，分别转换原生 B 样条和多项式；原跟踪样本单独保存，不代替完整未来轨迹。

`integrations/ros1/planner.py` 负责宿主侧运行适配，`worker.py` 负责 ROS 消息与进程生命周期，`launch.py` 从固定上游 XML/YAML 生成本次运行配置。部署仍由 `docker/ros1` 管理；worker 使用独立的最小源码包，不导入训练或 JAX 环境代码。

`SafeFlightCorridor` 和 `TrajectoryPreview` 各自保存生成时间、有效期和坐标系。它们属于规划检查数据，不参与控制或安全证明。决策归档保留每个模块的真实输出。回放只显示当前有效的数据，互不合并成一个可能缩短或延长有效期的总括对象。

## 安装、产物和兼容边界

配置版本为 4，原生 RPC 版本为 2。旧检查点不会在运行时隐式迁移。`artifacts/migration.py` 需要调用者提供审核后的 v4 解析配置，写入新文件并保留原始权重和元数据；输入或物理语义不匹配时拒绝迁移。

包内 `configs/`、`assets/`、`benchmarks/` 是唯一物理副本；源码根目录不再保留同名镜像或符号链接。安装代码使用 `importlib.resources`。输出由 `runtime.output_root` 指定，默认当前工作目录。训练可记录已安装依赖的版本和源码摘要，不要求依赖来自 Git 工作树。

普通 train/eval 不写完整回放；`evaluation.record_replays=true` 或 `play checkpoint=...` 才记录。工程接口验证与正式任务质量验证分开。正式选择和历史结果继续由原清单及原始报告确定，当前状态见 [status](status.md)，操作见 [runbook](runbook.md)。

## Racing 指标

Racing 验收检查合法门序、无失败完赛率，并报告完赛时间。参考轨迹误差仍用于控制诊断，不作为竞速质量门槛：合法、更快的路径可以偏离参考轨迹。Tracking 才以指定时间参考的 RMSE 作为质量条件。当前阈值和历史结果不因接入目录整理改变。
