# 学习模块

`train.py` 编排 Brax PPO/APG、BPTT、SHAC 和 D.VA。三个循环 BPTT 入口共用 `checkpointing.py` 的记录、选优和更新调度，保留各自的 rollout、loss、update 与恢复合同。网络位于 `../networks/`，冻结策略执行与重建位于 `inference.py`，训练环境包装在 `wrappers.py`。任务奖励在 `../environments/tasks/rewards.py`，学习侧轨迹和辅助预测损失在 `objectives/`。LOTF 只属于动力学，使用这些通用训练器。

恢复核对参数、优化器、随机数、预算和训练内选模历史。冻结评测复用 Task，不更新策略、优化器或归一化统计。完整预算、参数来源与策略质量分别记录，当前结论见[现役状态](../../../docs/status.md)。
