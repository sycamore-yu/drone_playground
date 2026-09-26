# 04：训练编排与LOTF BPTT

Type: task
Status: planned
Blocked by: 01, 03
Engineering: not-started
Experiment: not-started
Quality: not-started
Owner: unassigned

## 模块职责

训练配置只公开algorithm/network/objective/training四类；其它控制量作为对应内部参数。
LOTF的BPTT作为JAX作者适配，保留原网络、窗口、损失缩放及优化器；Brax APG保持独立配方身份。

## 实现证据

原生与适配后的短轨迹损失/梯度/更新对照，检查检查点重载和训练状态恢复。
全部有效配置解析后冻结，并保留作者机型/任务与Crazyflie实验的区别。

## 共同交付

悬停、八字两任务完成声明预算，输出回报/误差曲线、初始/中期/最终策略及实际耗时。
5秒在线适应等原论文结果属于本轮之外；低分和失败版本完整保留。
