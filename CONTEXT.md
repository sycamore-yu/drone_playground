# 领域术语

稳定术语在此维护。实现位置见 `docs/architecture.md`，当前活动见 `docs/status.md`。

| 术语 | 项目中的含义 |
|---|---|
| 实验配方 `experiment` | 使用 Hydra 组合环境、方法、训练算法、网络和运行设置 |
| 运行方法 `method` | 实际执行的 Policy、Planner、Controller 或已配置模块链 |
| 训练算法 `algorithm` | 更新参数的规则；PPO、BPTT、SHAC、DVA 属于训练算法 |
| 策略 `Policy` | 从观测产生已声明输出的可执行策略；循环记忆属于策略实例 |
| 规划器 `Planner` | 产生航点或时间轨迹的运行模块 |
| 控制器 `Controller` | 将输入参考或设定值变成其下游支持的物理控制量 |
| 环境 `Environment` | 直接组合 Dynamics、Controller、Reference、Scene、Sensor、Task |
| 任务 `Task` | 初始化、观测、成功/失败事件、时限、奖励或损失 |
| 场景 `Scene` | 物理资产、实例、运动参数及默认起终点；采样指令另行保存 |
| 传感器 `Sensor` | 在明确仿真时刻产生测量 |
| 观测 `Observation` | 输入字段、形状、单位、坐标和历史的规格及编码 |
| 参考 `Reference` | Waypoint 或 Trajectory；用于描述期望运动 |
| 航点 `Waypoint` | 有序空间目标，明确容差和有效期；不自带到达时间轨迹 |
| 轨迹 `Trajectory` | 带时间参数的分段期望运动，包含真实有效区间 |
| 状态设定值 `StateSetpoint` | 显式位置、速度、净加速度、yaw 或 yaw-rate 字段 |
| 姿态设定值 `AttitudeSetpoint` | 姿态弧度和总推力牛顿 |
| 角速度设定值 `RateSetpoint` | 机体系角速度弧度/秒和总推力牛顿 |
| 执行器输入 `Actuation` | ForceTorque 或 MotorRPM 原生物理输入 |
| 动力学 `Dynamics` | 通过 `step(state, control, dt)` 推进原生物理状态 |
| 预测模型 | 规划器或控制器内部使用的模型，与实际前向动力学分别声明 |
| 求导边界 | 梯度可传播、被截断或被显式替代导数定义的位置 |
| 安全飞行走廊 `SFC` | SafeFlightCorridor 规划检查数据，具有自身有效期 |
| 轨迹预览 `TrajectoryPreview` | 非执行轨迹的检查数据，具有自身有效期 |
| 人工截断 `truncation` | 为采样而停止；带 critic 的算法可使用重置前末态价值 |
| 任务终止 `termination` | 成功、失败或任务截止；不继续 bootstrap |
| 参数热启动 `warm start` | 从已有参数开始新训练，优化器与随机状态按新运行初始化 |
| 完整恢复 `resume` | 恢复参数、优化器、随机状态、计数及必要环境状态 |
| 冻结评测 `evaluation` | 固定参数与归一化统计执行任务并核对摘要 |
| 基准 `benchmark` | 版本化 cases、种子、预算、指标和判定规则 |
| 工程通过 | 对应实现、调用链和定向验证通过 |
| 质量通过 | 达到基准规定的成功率、误差或时间要求 |

运行角色只有 `train` 和 `eval`。训练内选模与正式 Benchmark 都使用 `eval`，由评测设置和协议区分。Figure-8、随机样条是 Tracking 的 reference presets；Static/Dynamic 是 Navigation 的场景集合。

Hydra/YAML 是唯一配置组合系统。`load()` 直接组合环境。项目不增加重复 registry、通用 DroneState、EnvironmentState 或 Manager。外层状态继续使用 Brax `State`，其内部保留后端和任务的原生数据。

环境统一持有 `env.freq` 和实际物理时钟；Task 不配置另一套频率。共享执行器管理动作到达、物理子步、检测结果累计和首次终止冻结，Task 定义事件规则。具体任务初始化、原生赛车判定和论文离散映射保持其已有语义。

动力学随机化在 reset 采样物理参数；测量噪声只改输入；动作误差改执行命令；外力/力矩属于运行扰动。初态、场景和指令分布分别声明。固定分布不称为 curriculum。

LOTF 只表示动力学来源。其 high-fidelity 与 simplified 实现不形成另一套 Task、Policy、Algorithm 或 Evaluation。物理数组顺序只在后端边界转换。

固定几何来自 MJCF。动态障碍当前位置保存在运行态。SFC、轨迹预览和可视化标记不作为传感输入或安全保证。配置、软件接口和工程测试迁移不修改历史权重、原始结果或正式质量结论。
