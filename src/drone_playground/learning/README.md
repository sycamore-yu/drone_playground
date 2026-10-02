# 学习模块

`train.py` 编排 Brax PPO/APG、BPTT、SHAC 和 D.VA；循环点云训练保留与真实积分、观测记忆相应的适配。网络位于 `../networks/`，训练环境包装在 `wrappers.py`，奖励／损失在 `objectives/`。LOTF 只属于动力学，使用这些通用训练器。

恢复核对参数、优化器、随机数、预算和训练内选模历史。冻结评测复用 Task，不更新策略、优化器或归一化统计。完整预算、参数来源与策略质量以[正式验收](../../../artifacts/verification/final-acceptance/README.md)的独立字段为准。
