# 控制器模块

`env.execution` 声明命令层级、可选轨迹跟踪器、内环及实际动力学。`trajectory.py` 将轨迹转换为姿态与总推力；`crazyflow.py` 接入姿态执行；`lotf.py` 复用 Betaflight 风格内环；`acceleration.py` 处理点云方法的加速度命令。

完整命令应用和物理子步顺序由上层 `execution/transition.py` 掌握，同一个控制阶段执行一次。MPC 的在线决策、预测模型与暖启动位于 `methods/optimal_control/`。轨迹跟踪层是本项目控制比较的主要替换位置，内环保持独立职责。

`demo.py` 提供既有 Mellinger 固定示例。各配方的真实支持能力、调用频率和数值验证记录见 [架构](../../../../docs/architecture.md)。
