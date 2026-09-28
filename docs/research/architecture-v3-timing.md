# 时序与延迟依据

2026-09-28 核对。区分执行器一阶响应、动作/观测队列，以及将实测计算耗时注入仿真三类行为。

## 上游支持的事实

- FlightBench 论文第 IV-A、IV-D 节说明 ROS 通信形成约 45 ms 的系统延迟，并在策略训练中采用 25–50 ms 的延迟随机化。来源：https://arxiv.org/html/2406.05687v3 。
- Isaac Lab 提供 DelayedPDActuator，其控制目标可进入每环境独立延迟队列，范围由配置选择。框架能力与具体任务启用范围分别理解。来源：https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html 。
- MuJoCo Playground 的 Franka 推方块任务使用动作及观测历史队列和随机延迟。这是具名任务行为。来源：https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/_src/manipulation/franka_emika_panda_robotiq/push_cube.py 。
- DiffAero 的 pointmass 动力学包含响应滞后。核对源码提交 291ea14196aefbebcf7387dd71f7e096c83878b7；`dynamics/pointmass.py:120–148,214–218` 的控制响应系数由指数函数给出，表示一阶响应滞后。来源：https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/dynamics/pointmass.py 。

## 本项目决定

迁移回归保持同步仿真和零附加延迟，已有具名执行器/动作响应继续保留。新增显式 `runtime.action_delay_steps` 用于需要该时序条件的配方，队列按环境重置；测试验证因果顺序和重置隔离。计算时间独立统计，当前默认协议按仿真时钟推进。

原生 ROS 冷启动独立于飞行时间。并发验证中 SUPER 建图初始化超过旧 20 s 就绪上限，日志表明工作进程仍处于 A* 地图初始化。就绪等待上限调整为 90 s，通信等待保留额外裕度；失败运行及修复后的新运行分别保留。

当前传感器改造聚焦已有现实测量特征的正确实现和来源记录；传感器变化消融作为单独研究范围处理。

本轮恢复时已重新打开 FlightBench 论文与两套框架官方文档/源码核对上述位置；DiffAero 的同提交本地源码也复核了连续与离散状态更新。框架提供某项延迟能力与默认任务实际启用该能力分别记录。
