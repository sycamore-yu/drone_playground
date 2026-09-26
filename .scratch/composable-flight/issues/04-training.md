# 04：训练编排与LOTF BPTT

Type: task
Status: resolved
Blocked by: 01, 03
Engineering: passed
Experiment: completed
Quality: passed
Owner: current-chat

## 模块职责

训练配置只公开algorithm/network/objective/training四类；其它控制量作为对应内部参数。
LOTF的BPTT作为JAX作者适配，保留原网络、窗口、损失缩放及优化器；Brax APG保持独立配方身份。

## 实现证据

原生与适配后的短轨迹损失/梯度/更新对照，检查检查点重载和训练状态恢复。
全部有效配置解析后冻结，并保留作者机型/任务与Crazyflie实验的区别。

## 共同交付

悬停、八字两任务完成声明预算，输出回报/误差曲线、初始/中期/最终策略及实际耗时。
5秒在线适应等原论文结果属于本轮之外；低分和失败版本完整保留。

## Comments

2026-09-26：用户确认四类训练设置；Notebook核对为悬停600万、八字2250万次交互，前向按本轮范围切为高保真，其余主要配方沿原实现。

## Answer

四类训练配置、Brax PPO/APG/SHAC和原生LOTF BPTT已接到共同入口。
两任务分别完整完成200/300更新、600万/2250万交互，各保存9个初始/中间/最终策略。
原生单次loss、随机数和Adam更新对照通过。续训保存优化器、环境、随机数与恢复时刻以前的
开发最佳模型，来源独立记录，后续时刻评估排除在恢复选模之外。
CPU小规模续训逐元素一致；正式GPU175→200更新的最大参数差4.59e-6，开发32/32仍完成。
恢复精度作为实测边界保留，见[恢复核验](../../../docs/verification/composable-lotf-resume-check.json)。
