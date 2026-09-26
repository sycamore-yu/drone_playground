# 02：策略、轨迹规划与统一控制器

Type: task
Status: resolved
Blocked by: 01
Engineering: passed
Experiment: completed
Quality: passed
Owner: current-chat

## 模块职责

策略输出轨迹或直接命令；轨迹生成、读取和在线规划属于其轨迹规划实现。
控制器对外一个预设，内部包含必要的跟踪、飞控和混控。保留原MPC、Crazyflow与LOTF调用语义。

## 实现证据

固定样条→MPC→姿态内环、直接姿态→拟合动力学、角速度/推力→LOTF控制器三类组合。
对照原生动作、单位、频率和内部状态；直通模式只在动作契约匹配时允许。

## 共同交付

LOTF策略和控制链与03–05共享接口，训练后能在同一前向组合中重评。

## Comments

2026-09-26：用户确认统一控制器、策略含轨迹规划和动作契约；与其余模块共同完成并交付训练结果。

## Answer

固定八字、随机样条、LSY赛道和作者CSV各有轨迹规划实现；冻结神经策略由共同入口加载。
统一controller预设接入Crazyflow姿态链、真实acados和采样MPC，以及LOTF原生飞控。
组合入口后的两MPC分别完成857/850步真实执行和5次原生过门；LOTF每子步仅调用一次控制器。
12格旧任务固定输入输出回归、原生控制输出及动作契约验证见
[工程记录](../../../docs/verification/composable-lotf-engineering.md)。
