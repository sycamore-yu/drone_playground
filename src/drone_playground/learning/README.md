# 学习模块

`train.py` 编排 Brax PPO/APG 与具名训练器；`algorithms/` 保存 SHAC、D.VA、LOTF 和点云时间反传的更新规则。网络实现在同级 `../networks/`，环境包装在 `env_adapter.py`，具名任务损失在 `objectives/`。任务奖励沿用已核对的任务实现及上游，算法负责将训练信号组合为更新损失。

PPO 支持参数暖启动；SHAC、D.VA、LOTF 与点云具名训练器按其能力保存完整训练状态。恢复核对参数、优化器、随机数、行为配置和开发集选择历史。参数热启动与完整续训分别记录。

本轮五任务 PPO 各 256 交互、点云 2 次更新，用于验证真实更新、保存和重载。完整训练预算及策略质量以 [本次验证](../../../docs/verification/architecture-v3/README.md) 的独立字段为准。LOTF 与点云论文保持独立方法身份。
