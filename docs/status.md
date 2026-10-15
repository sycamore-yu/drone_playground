# 当前状态

本文件记录 `feat/ppo-stability-acceptance-20261015` 的工作状态；修改尚未合入
`v0.2`。本轮按用户明确要求验证三次连续完整检查、Navigation 续训能力保持与
独立冻结测试，不改写历史单次完整验证协议。

- Tracking 的 PPO/APG/SHAC 三格已用原始 CSV、模型哈希和选模记录重新核对，
  三个独立随机种子均满足原 C1–C6；当前共核对通过 3/18 格。
- Navigation 的 Depth/LiDAR PPO 64 环境候选仍在训练；两者在 1,105,920 次交互
  的完整开发筛查均为 0/48，尚无稳定收敛证据。Depth APG/SHAC 正在相同条件下
  从零训练；两者在同一交互数的完整筛查也为 0/48。正式三种子验收仍待完成。
- Depth 奖励分叉对照完成 368,640 次额外交互：原奖励末次 0/48，最小间距奖励
  末次 4/48，后者仅保留 4/21 个原成功案例，未达到晋级要求。仅训练更新采用
  确定性 GPU 编译的分支已通过完整源模型复放，续训能力保持仍在验证。
- 已从冻结 Depth PPO 新采集训练成功回合，用于独立策略 KL 约束的短程诊断。
  原策略序列重算、零权重及正权重完整更新检查通过，180 次续训对照正在运行。
  两个冻结策略的 256 回合 Critic 校准已完成：固定训练状态上出现实际成功率下降、
  价值预测反而升高的情况。价值误差是否导致遗忘仍待受控验证。
- Racing 正在核对统一 progress=1、额外 failure cost=0 的任务奖励。SHAC seed 0
  连续三次完整检查达标，自动开发 benchmark 为 100/100；PPO 已耗尽 67,108,864
  次交互预算，末次 0/100；APG 在相同交互预算结束时也为 0/100。
  其余种子及最终独立冻结测试尚未完成，不能与其他奖励条件的旧成绩拼接。
- Navigation 的三种子验收、至少 900,000 次额外交互的能力保持及冻结 benchmark
  尚未完成。当前没有外部阻塞。

原始证据与复算脚本位于
`artifacts/verification/ppo-stability-20261015/`，训练产物位于
`results/goal_reaching_batch64/`、`results/goal_reaching_differentiable/` 与
`results/racing_common_reward_p1f0/`。
研究过程放在 `notes/ppo-stability-20261015.md`，不构成验收结果。
