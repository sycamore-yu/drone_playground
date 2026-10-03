# 可组合无人机环境

公开入口使用 Hydra 组合和实例化。项目不建立 Environment、Dynamics、Sensor、Controller 或 Algorithm registry。`drone_playground.load("hovering")` 只加载环境，不选择 PPO 或其他训练配方。

## 配置与构造

`environments/environment.py` 解析运行条件并创建六个组件。`environments/base.py` 的 `DroneEnvironment` 接收组件对象。

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

每个控制间隔执行输入转换、传输延迟、剩余控制阶段和物理推进，再计算观测、任务事件和奖励。导航在物理子步保留首次碰撞证据。控制频率必须与实际物理时钟匹配；不能仅修改一个标签便改变后端实际频率。

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

外部方法分为三层：`rpc/` 只定义 transport 和 wire contract；
`integrations/` 是随 Python 包安装的进程/ROS 适配器；仓库根目录
`native/` 保存 C++ SDK、Docker、ROS 补丁等非 Python 构建与部署文件。
这里所说的 native bridge 是后两层把独立进程的生命周期、状态/传感输入和
Reference/Setpoint 输出接回公共 Pipeline 的适配边界，不是另一套 Planner API。
进程内 JAX/Flax 方法（例如未来直接实现为 Policy 的 AllocateNet）不经过 bridge；
只有选择独立 C++/ROS/其他运行时版本时才需要对应 adapter。

低频输出只能在其原有效期内复用。缓存命中不刷新生成时间。无解或过期结果不再传给下游。重置清空模块缓存、调用计数、循环记忆和求解器历史。一次失败不会重置批次中仍在运行的其他实例。

## 观测与训练

观测组件声明字段、形状、单位和坐标。已有状态与感知网络继续使用原有向量布局，保留权重树；循环加速度环境的公共观测使用 `state`、`sensor`、`valid` 字典。自动重置按 pytree 操作，不假定观测一定是一个数组。

检查点保存解析配置、输入规格、动作维度、物理解码合同和参数摘要。恢复不能把相同宽度的另一组输入解释为原输入。完整恢复还保存优化器、随机键、计数及必要环境状态。PPO 参数热启动与完整优化器恢复分别记录。

训练创建 `train` 环境，训练内选模和最终冻结评测创建 `eval` 环境。运行角色只有这两个。训练分布、初态随机化、测量噪声、动作误差和动力学随机化在构造前解析；名义评测不继承训练扰动。

`termination` 与 `truncation` 保留不同语义。有 critic 的算法用重置前末态处理人工截断；真实终止不继续 bootstrap。奖励公式、参数及首次事件顺序不因架构迁移改变。

## 场景资产与回放

固定几何的物理源位于 `assets/scenes/` 的 MJCF。Navigation8 仍是 S01/S02/S03/S06、D01/D02/D03/D06；`catalog.xml` 组织场景和非几何元数据，独立场景 XML 保存实际图元，`boundary.xml` 保存共同边界。运动参数存于模型数据，当前位置按仿真时钟计算并写入运行态。

传感、净空和碰撞查询使用从编译模型提取的 `SceneBank` 数组。固定场景回放直接附加同一 MJCF；程序生成的训练场景由场景模块序列化其已采样参数。可视化模块不另造一份障碍几何。

LSY 赛道的位置、形状、门顺序和初态来自 `lsy_level0.xml`。原生任务代码继续处理过门和接触规则。非几何上游参数由 YAML 明确提供。Crazyflie 2.x 回放模型也作为独立 MJCF 资源安装。

`benchmarks/navigation-mjcf-verification.json` 校验当前 XML 摘要，并保留原目录的内容摘要。迁移不改变原 Navigation8 cases、原质量判定或历史报告。正式几何变更需要新验证记录。

## 外部接入与检查数据

Python/C++ 共用 `rpc/proto/algorithm.proto` 的 v2 协议。初始化、重置、一步调用、输出有效期和关闭均明确。EGO-Planner 与 SUPER 保持原 ROS 节点，分别转换原生 B 样条和多项式；原跟踪样本单独保存，不代替完整未来轨迹。

`SafeFlightCorridor` 和 `TrajectoryPreview` 各自保存生成时间、有效期和坐标系。它们属于规划检查数据，不参与控制或安全证明。决策归档保留每个模块的真实输出。回放只显示当前有效的数据，互不合并成一个可能缩短或延长有效期的总括对象。

## 安装、产物和兼容边界

配置版本为 4，原生 RPC 版本为 2。旧检查点不会在运行时隐式迁移。`artifacts/migration.py` 需要调用者提供审核后的 v4 解析配置，写入新文件并保留原始权重和元数据；输入或物理语义不匹配时拒绝迁移。

包内 `configs/`、`assets/`、`benchmarks/` 是唯一物理副本；源码根目录不再保留同名镜像或符号链接。安装代码使用 `importlib.resources`。输出由 `runtime.output_root` 指定，默认当前工作目录。训练可记录已安装依赖的版本和源码摘要，不要求依赖来自 Git 工作树。

普通 train/eval 不写完整回放；`evaluation.record_replays=true` 或 `play checkpoint=...` 才记录。工程接口验证与正式任务质量验证分开。正式选择和历史结果继续由原清单及原始报告确定，当前状态见 [status](status.md)，操作见 [runbook](runbook.md)。
