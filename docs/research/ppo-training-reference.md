# PPO 调参参考与当前约束

2026-10-09 核对旧仓库及 DiffAero 固定源码版本 `291ea14196aefbebcf7387dd71f7e096c83878b7`。以下是候选依据，不能作为新仓库训练成功的证据。

DiffAero 的 [AsymmetricPPO](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/algo/PPO.py)分别缓存 Actor 测量观测和 Critic 仿真状态，并从 reset 前的下一状态计算值函数。它的 [障碍规避状态和奖励](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/env/obstacle_avoidance.py)包括目标距离、姿态、速度和所有障碍最近点；规避项会惩罚接近危险表面，并奖励危险附近的横向避让。当前实现借鉴输入分离，使用固定全向理想射线表示真实几何；不声称与其最近点 Critic 完全等价。

DiffAero [当前 OA 默认奖励](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/cfg/env/oa.yaml)标注为 SHA2C，其前一版 PPO 权重以注释保存。不能把当前默认权重当作经过本项目 PPO 验证的配方。DiffAero 目标包含到达后悬停，本项目 Navigation 到达即终止；直接增加每步正存活奖励可能让延迟到达更划算。因此先使用已实现的目标进度奖励，并保留现有高度、净空和失败代价。

旧仓库的 Navigation PPO 含 `clip_epsilon=0.1`、熵权重 `0.001`、`gamma=0.995` 与进度奖励；参考配置采用不同控制接口和整体奖励缩放。其 PointNet/GRU192 开发候选采用 batch32/horizon32/lr3e-4/净空 margin1.5，主场景 47/48，仍有未达标场景，也未完成三种子正式验收。较大 horizon 与较小 Actor 输出初始化均曾降低该开发成绩。

本批次的候选筛选只读取完整 checkpoint 的固定六个主场景（150 回合），同时报告最差场景及 C5 连续通过数。独立冻结不用于调参。`lr`、`clip_epsilon`、熵、折扣、GAE、进度权重和速度窗口使用现有参数，范围保存在 `.optim-agent-runs/experiment.json`；特权输入边界见 [ADR-0010](../adr/0010-training-only-privileged-ppo-critic.md)。

Depth 低学习率已经降低 KL，但至第 1120 次仍全零，说明更新幅度不足以解释全部失败。进度奖励 0.5 的独立对照仍在观察。LiDAR 同观测独立 Critic 第 800 次已升至 63/150，说明不能在前几个全零检查后断言永远无效。另一待测现有参数是 `velocity_window=1`，用更即时的速度目标缓解约三秒历史平均与短 GAE 的信用分配差异。没有证明任何一个候选必然收敛。
