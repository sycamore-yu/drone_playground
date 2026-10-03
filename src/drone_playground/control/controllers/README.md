# 控制器模块

`env.controller` 声明环境实际执行的控制边界。`trajectory.py` 将 Trajectory / Waypoint 降为姿态与总推力；`crazyflow.py` 处理姿态执行；`bodyrates.py` 声明总推力／机体系角速度接口，LOTF 在物理推进时执行其原生 Betaflight 内环；`acceleration.py` 处理加速度型 PointMass 任务。

完整命令应用和物理子步顺序由 `control/transition.py` 掌握。MPC 的在线决策、预测模型与暖启动位于 `control/controllers/mpc/`。轨迹跟踪层是控制比较的主要替换位置，内环和实际 Dynamics 保持独立职责。

公共输入类型定义在 `references.py` 与 `control/setpoints.py`。各配方的真实支持能力、调用频率和数值验证记录见 [架构](../../../../docs/architecture.md)。
