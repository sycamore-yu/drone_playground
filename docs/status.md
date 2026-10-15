# 当前状态

本文件记录 `feat/ppo-stability-acceptance-20261015` 的工作状态；修改尚未合入
`v0.2`。本轮按用户明确要求验证三次连续完整检查、Navigation 续训能力保持与
独立冻结测试，不改写历史单次完整验证协议。用户已明确允许不同算法使用不同
训练奖励，只要任务目标一致；已有从零训练证据可以复用，不要求全部在当前代码重训。

- 直接复用已有核对后，当前矩阵通过 7/18 格：Tracking 三算法、Racing PPO/BPTT、
  Depth BPTT 静态/动态。18 个选定运行均复用已有逐回合、选模和模型完整性结果，
  本次未重跑这些验收。Depth BPTT 使用旧 `scratch_depth_apg_s0/1/2` 的完整审计。
- 历史 SHAC 与 LiDAR BPTT 均有主场景达标结果；部分种子按单次完整验证协议
  选模，尚缺本轮三次连续检查。保存于 `.scheduler/frozen_audits/` 的结果也纳入
  缺口分析，不能仅因训练目录写着 interrupted 就判定没有通过。
- Navigation 的 Depth/LiDAR PPO 和同条件的 Depth APG/SHAC 均完成 2,211,840 次
  交互，末次完整开发筛查均为 0/48，完整训练恢复点已保留。没有稳定收敛证据，
  正式三种子验收仍待完成。
- Depth 奖励分叉对照完成 368,640 次额外交互：原奖励末次 0/48，最小间距奖励
  末次 4/48，后者仅保留 4/21 个原成功案例，未达到晋级要求。仅训练更新采用
  确定性 GPU 编译的分支也完成同一预算，末次 1/48、保留 0/21，不能防止退化。
- 训练成功经历的策略 KL 约束诊断已完成：首轮 22/48、保留 18/21，末次降为
  4/48、保留 3/21，未通过预设门槛，不将该候选叠加到下一配方。
- 完整回报校准与原轨迹复放确认：在这批训练状态上，512 步、GAE λ=1 的 target
  偏差明显小于 λ=0.95。两条相同长 rollout、仅 λ 不同的续训对照正在执行；
  已通过原状态复放和实际 GPU 更新检查，能力保持收益仍待验证。
  首次完整检查 λ=1 为 28/48、保留 18/21，λ=0.95 为 24/48、保留 14/21；
  λ=0.95 第二次降为 6/48、保留 5/21，末次 4/48，已完成；λ=1 第二次 28/48，
  仍在完成末次检查，不能据此晋级。
- Racing 正在核对统一 progress=1、额外 failure cost=0 的任务奖励。SHAC seed 0
  连续三次完整检查达标，自动开发 benchmark 为 100/100；PPO 已耗尽 67,108,864
  次交互预算，末次 0/100；APG 在相同交互预算结束时也为 0/100。
  APG 低学习率对照末次仍为 0/100；随后增加 horizon 的试验已按用户澄清停止，
  保留完整恢复点，不再为统一奖励重复训练已有三次连续达标的 Racing BPTT。
  PPO 的参考优化参数组已完成三个独立随机种子及预留 4million 批次的冻结测试，
  各 100/100；原始 CSV、实际加载模型及六份回放已核对，逐种子 C1/C3/C5/C6
  通过。SHAC seed 1/2 的既有试验仍在执行，已有 seed 0 三次连续达标；继续的
  依据是连续检查缺口。不同奖励的任务达标证据允许进入各算法单元，并披露差异。
- Navigation 的三种子验收、至少 900,000 次额外交互的能力保持及冻结 benchmark
  尚未完成。当前没有外部阻塞。

原始证据与复算脚本位于
`artifacts/verification/ppo-stability-20261015/`，训练产物位于
`results/goal_reaching_batch64/`、`results/goal_reaching_differentiable/` 与
`results/racing_common_reward_p1f0/`。
研究过程放在 `notes/ppo-stability-20261015.md`，不构成验收结果。
当前正式证据可由 `tools/report_acceptance.py` 读取
`artifacts/verification/ppo-stability-20261015/acceptance-manifest.json`，以
`--output-prefix artifacts/verification/ppo-stability-20261015/acceptance-reused` 重建；
其矩阵中缺失的正式种子不以开发试验代替。
