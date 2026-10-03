# ADR 0004：Reference、Setpoint 与原生动力学推进

## Status

Accepted. Supersedes ADR 0001.

## Context

同一个四元素数组曾同时承载姿态/推力和推力/角速度。动力学外部入口也不一致。宿主规划检查数据共用一个有效期会丢失不同走廊和预览的时间语义。

## Decision

公共参考使用 Waypoint/Trajectory。控制输入使用 StateSetpoint、AttitudeSetpoint、RateSetpoint；原生执行器输入使用 ForceTorque/MotorRPM。每个类型明确物理字段和单位。删除扁平 MotionCommand/COMMANDS 体系。

Dynamics 使用 `step(state, control, dt)`。输入必须与执行控制器输出一致，状态保留后端原生类型，dt 使用秒。接口不替换 Crazyflow、LOTF 或 PointMassLag 的原有积分与内环。低层输入只运行剩余控制阶段。

Reference 的 NumPy 和 JAX 表示共用数学与字段；JAX 可追踪构造、批处理和梯度。外部 RPC 的可微能力由服务单独声明。C++ 或 ROS 接入不自动获得可微能力。

SafeFlightCorridor 与 TrajectoryPreview 各自保存生成时间和有效期。执行输出与检查数据分别存储。固定几何以 MJCF 为源，运行态承担动态位姿，回放使用同一资产。

## Consequences

后端需要显式转换原生数组顺序。Crazyflow 的角速度顺序为 `[wx, wy, wz, T]`，LOTF 为 `[T, wx, wy, wz]`。参数宽度不能充当类型检查。

配置升级至 v4，RPC 升级至 v2。历史检查点只能经显式复制迁移；旧原生服务必须使用对应 v2 SDK 重新构建。原始实验包、报告和权重保持原样。

验证覆盖类型/时间/梯度、原生推进一致性、实际训练与恢复、MPC、C++/ROS、MJCF 与回放。工程验证不替代完整任务质量评测。
