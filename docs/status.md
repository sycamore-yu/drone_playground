# 当前状态

本文件记录 `feat/ppo-stability-acceptance-20261015` 的工作状态；修改尚未合入
`v0.2`。本轮按用户明确要求验证三次连续完整检查、Navigation 续训能力保持与
独立冻结测试，不改写历史单次完整验证协议。

- Tracking 的 PPO/APG/SHAC 三格已用原始 CSV、模型哈希和选模记录重新核对，
  三个独立随机种子均满足原 C1–C6；当前共核对通过 3/18 格。
- Navigation 的 Depth/LiDAR PPO 正在运行 16/64 环境的从零对照。新任务奖励使用有界代价、
  安全到达终端奖励、无折扣回报和有限时域；实现检查通过，尚无收敛证据。
  APG/SHAC 已支持相同距离课程，仍需在选定共享任务条件下完成正式训练。
- Racing/SHAC seed 0 在共享奖励下连续三次完整检查达标，新的独立冻结测试
  100/100 合法完赛；其余种子与 APG 尚待补验。共享奖励下的 PPO 候选持续
  未过第一道门，已保存恢复点并结束该候选。历史三个算法的奖励系数不同，
  不能直接拼成同条件矩阵。
- Navigation 的三种子验收、至少 900,000 次额外交互的能力保持及冻结 benchmark
  尚未完成。当前没有外部阻塞。

原始证据与复算脚本位于
`artifacts/verification/ppo-stability-20261015/`，训练产物位于
`results/goal_reaching_pilot/` 与 `results/racing_common_reward/`。
研究过程放在 `notes/ppo-stability-20261015.md`，不构成验收结果。
