# ADR-0015：三阶段学习收敛验收与 GPU 吞吐测量

**Status:** Accepted · 2026-10-10

## Context

先前 C5 要求每种子连续三次 **完整** checkpoint evaluation 达标。Navigation 的每次检查包括六个主场景及 S06/D06，各场景 25 回合（200 回合）。Depth/LiDAR 一次评测实测约需 110–140 秒。它造成反复完整评测的成本，并让三个独立种子在有限样本的过严连续门槛下反复训练。

用户批准保留 **18 个验收单元、每配方 3 个随机初始化训练种子、C2–C4 原有任务质量门槛、独立冻结 Benchmark**，只改变中间诊断／选模协议；任何旧项目权重不用于本轮初始化。

## Decision

新训练选模规则为 `learning.selection=first_full_validation_pass`，使用单一训练器与环境：

1. **训练中轻量筛查（screening）**：每次 `learning.evaluation_interval`，只评估 Navigation 六个主场景，各 **8 回合**；Tracking/Racing 评估对应场景 **8 回合**。筛查采用 `diagnostic` 协议，不记作 C2/C3/C4 达标。用于触发正式选模的宽松条件：Navigation 每主场景至少 6/8；Tracking 至少 7/8 且成功回合位置 RMSE ≤0.35 m；Racing 至少 7/8。新运行的筛查记录放在 `checkpoints/model_*/screening/`；v2 历史 run 仍在 `checkpoints/step-*/screening/`。
2. **完整候选选模（checkpoint validation）**：只对通过筛查的候选执行一次完整检查：Navigation 六个主场景各 **25 回合**；Tracking/Racing 各 **100 回合**。继续使用 C2/C3/C4 的硬门槛。筛查与正式选模采用不同的随机种子基段：筛查 `1_000_000`，完整选模 `1_500_000`；二者均区别于训练随机种子。首次完整通过者作为固定的候选权重，不要求连续三次完整评测。
3. **独立冻结 Benchmark**：停止更新，固定上述候选，以 **≥2_000_000** 的独立种子基段做一次正式冻结评测。Navigation 六个主场景各 25 回合，另对 S06/D06 各 25 回合完整记录（扩展场景不作 C4 门槛）；Tracking/Racing 各 100 回合。最终是否通过仍由 C2/C3/C4 逐场景硬阈值决定。冻结 Benchmark 失败不反复使用相同测试集来挑选权重，应另开具名新训练试验。

新 v3 run 的 `checkpoints/model_NNNNNN/training/` 使用 Orbax 保存完整训练状态，在该模型首次正式通过时向其 `actor/` 导出推理策略，详见 [ADR-0016](0016-results-layout-orbax-tensorboard.md)。历史 run 保留 `latest.training.zip`，不为每次筛查复制推理策略归档。筛查与完整选模可在同一次训练更新后连续发生，但不能共用评测种子。

## Migration of running experiments

启动前已有的 `last_of_first_three_consecutive_passes` 训练保留原始规格及三次连续结果，不重写历史评分；同一 run 的续训必须显式保留旧 `learning.selection` 与 `required_consecutive_passes=3`，否则严格 checkpoint 合同不允许恢复。新增训练默认三阶段协议及 `required_consecutive_passes=1`。不同协议的成绩标注规则，不能假装完全相同的选模成本。

2026-10-15 用户在 PPO 稳定性任务中再次选择本协议，并明确允许复用各历史源码
版本的从零收敛证据、允许算法间不同训练奖励，任务目标与阈值保持一致。
已完成的旧连续三次验证、以及 `.scheduler/frozen_audits/` 中一次性首个完整
候选冻结审计可以纳入矩阵；不改写原配置或选模记录。PPO Navigation 额外保留
至少 900,000 次交互的续训稳定性窗口和预定的原成功案例保持门槛。
本轮汇总器已支持这些记录；训练算法的稳定收敛仍以实测结果判断。
本轮实现核对时间：`2026-10-15T14:58:15Z`，分支实现提交 `9cc19e3`；
这次核对不改变原 ADR 的批准时间，也不表示已经合入 `v0.2`。

## GPU performance

保留 `XLA_PYTHON_CLIENT_PREALLOCATE=false`；GPU 活跃率和显存使用率不当成训练效率的唯一目标。对照真实 Crazyflow 训练，分别在稳定预热后测量：

- 真实每次参数更新耗时、每秒交互数、显存峰值与 GPU 功率；
- `num_envs` 变化对处理量、单次更新样本数和到达质量阈值总时间的影响；
- 感知传感器的独立额外开销，以及与选模／冻结评测成本的占比。

**当已有训练占用 RTX 4090 时，不启动同时竞争 GPU 的吞吐基准**。吞吐对照在 GPU 空闲时单独执行，独立存放在 Git 忽略的 `results/`，不将不同机器与任务间的 FPS 直接当作算法优劣。

## Relation to other acceptance criteria

2026-10-10 的 [ADR-0017](0017-eight-acceptance-gates.md) 将旧 C1–C5 的
完整种子/任务门槛纳入新 L3，将旧 C6 与 L6 的资源报告纳入新 L4；
独立平台合同合并为 S1–S4。数值门槛不变，三阶段评测的实现仍归本 ADR，
不再将旧 20 项清单作为额外完成任务。
