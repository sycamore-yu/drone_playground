# 当前状态

本文件记录 `feat/ppo-stability-acceptance-20261015` 的工作状态；修改尚未合入
`v0.2`。本轮按用户明确要求验证三次连续完整检查、Navigation 续训能力保持与
独立冻结测试，不改写历史单次完整验证协议。

- Tracking 的 PPO/APG/SHAC 三格已用原始 CSV、模型哈希和选模记录重新核对，
  三个独立随机种子均满足原 C1–C6；当前共核对通过 3/18 格。
- Navigation 的 Depth/LiDAR PPO 64 环境候选仍在训练；两者在 1,105,920 次交互
  的完整开发筛查均为 0/48，尚无稳定收敛证据。Depth APG/SHAC 正在相同条件下
  从零训练，正式三种子验收仍待完成。
- Depth 从同一完整状态分叉的奖励与确定性 GPU 续训诊断正在进行。GPU 首次更新
  重复性检查已验证确定性设置能消除该检查中的参数差异；它是否改善长期能力保持
  尚未证实。最小间距奖励的首轮检查只保留 10/21 个原成功案例，不能据此晋级。
- Racing 正在核对统一 progress=1、额外 failure cost=0 的任务奖励。SHAC seed 0
  连续三次完整检查达标，自动开发 benchmark 为 100/100；PPO/APG 对照仍在训练。
  其余种子及最终独立冻结测试尚未完成，不能与其他奖励条件的旧成绩拼接。
- Navigation 的三种子验收、至少 900,000 次额外交互的能力保持及冻结 benchmark
  尚未完成。当前没有外部阻塞。

原始证据与复算脚本位于
`artifacts/verification/ppo-stability-20261015/`，训练产物位于
`results/goal_reaching_batch64/`、`results/goal_reaching_differentiable/` 与
`results/racing_common_reward_p1f0/`。
研究过程放在 `notes/ppo-stability-20261015.md`，不构成验收结果。
