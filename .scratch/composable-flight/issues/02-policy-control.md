# 02：策略、轨迹规划与统一控制器

Type: task
Status: planned
Blocked by: 01
Engineering: not-started
Experiment: not-started
Quality: not-started
Owner: unassigned

## 模块职责

策略输出轨迹或直接命令；轨迹生成、读取和在线规划属于其轨迹规划实现。
控制器对外一个预设，内部包含必要的跟踪、飞控和混控。保留原MPC、Crazyflow与LOTF调用语义。

## 实现证据

固定样条→MPC→姿态内环、直接姿态→拟合动力学、角速度/推力→LOTF控制器三类组合。
对照原生动作、单位、频率和内部状态；直通模式只在动作契约匹配时允许。

## 共同交付

LOTF策略和控制链与03–05共享接口，训练后能在同一前向组合中重评。
