# ADR-0018：训练诊断 Patience 与一次性冻结选模

**Status:** Accepted · 2026-10-10

**后续细化（2026-10-14）：**Navigation PPO 的**新训练**早停增加
真实进度、失败前运动、高度稳定性及价值校准诊断；不能仅靠持续
`0/48` 判断“没有学习”。已有能力的后续退化必须保留最佳权重并单独
记录。详见 [ADR-0024](0024-navigation-ppo-progress-regression-screening.md)。
历史训练仍按各自保存的配置执行，此补充**尚未部署到正在运行的调度器**。

## Decision

新的随机初始化训练遵循 [ADR-0015](0015-three-stage-learning-acceptance.md)：

1. 使用轻量筛查判断训练进展，而不是在训练中反复运行 200 回合完整冻结测试。
2. **首次完整 checkpoint validation 达到 L3 的正式门槛**，立即选择此权重并运行
   **一次**独立冻结 Benchmark。通过则终止该种子并记成功；失败则保存原结果并
   新建具名随机初始化试验，不在相同的冻结种子上反复挑选多个 checkpoint。
3. 新训练在 **至少 800 次参数更新**之后，连续 **6 次**轻量筛查没有显著改善时，
   以 `early_stopped` 结束当前配方。进展发生时 Patience 清零，训练继续。
4. 轻量筛查的进展度量只使用每主场景 **8 回合**的诊断数据：
   Navigation 采用 `最差主场景成功率 + 0.25 × 六场景平均成功率`；
   Tracking/Racing 采用主场景成功率，Tracking 加上截断至 [0,1] 的成功 RMSE 奖励。
   `min_delta=0.02` 为显著改善的最小增量；诊断结果不能代替正式 L3 通过标准。
5. Patience 从最小更新数达到之后才累计，无论中途怎样改进，都保留最优
   **训练诊断**分数、连续停滞次数、结束原因和完整训练状态。
6. 调度器不再对新的三阶段配方机械地每 1,000 updates 发送停止信号。
   旧版 v2 的预算机制只作为历史训练的资源防护，不能把它当成 Patience。

## Existing historical runs

在 2026-10-10 前启动的 v2 ZIP 训练可能已经有一次或多次完整 checkpoint
validation 达标，但没有三连通过。**不改写它们的原始规则和 ZIP 参数**。
如需立刻节省 GPU 时间，仅允许对该 run 中**最早的一次完整通过**的
`step-*/policy.zip` 执行一次独立冻结 Benchmark（种子基段 ≥2,000,000）。
该旧版候选必须先安全停止当前训练再验证。

一次性审查工具：`tools/freeze_legacy_candidate.py`。
实际结果保存在 `results/scratch_v02/.scheduler/frozen_audits/<run_name>/`，
其中 `audit.json` 固定候选路径和结果状态，`report.json`、
`episodes.csv` 保存完整冻结分母。若该目录已存在则拒绝再次运行，
避免重复使用同一冻结测试挑选权重。成功的旧训练种子可作为同一份
**从零训练证据**纳入矩阵，新种子仍须完全独立初始化。

冻结失败的新实验不得把测试种子上的具体场景成绩当作超参选择信号。
失败后的重启和新配方仍以**训练诊断/完整 checkpoint validation**为依据。
独立验证及历史档案不得用于训练梯度或奖励优化。

## Results and algorithms

新运行的 `config.yaml` 在顶层 `early_stopping` 记录以上参数，
其位置不改变旧版 `learning` 优化器/梯度数值合同，保留严格续训能力。
在 TensorBoard 与 `run.json` 中记录轻量筛查评分、最优评分、
停滞计数和 early-stop 事件。停止后保留 Orbax 模型与完整优化器状态，
并标记不可因无进展而自动续训。所有 L3 原始通过率和三独立种子要求不变。
