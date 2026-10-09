# 实验配方

当前唯一权威配置是 `src/drone_playground/configs/` 的 Hydra 配方。Task 定义目标与判定，
`method.action` 定义动作单位，`controller` 决定轨迹跟踪，`learning` 决定算法、损失和
反向模型。新增研究方法应直接复用这些配置组，不建立第二份 Registry 或默认值副本。

## 新实验

```bash
pixi install --locked
pixi run train experiment=tracking learning=apg seed=0 \
  output=results/tracking_apg_new_s0
pixi run train experiment=tracking_point_mass learning=apg seed=0 \
  output=results/tracking_point_mass_new_s0
pixi run train experiment=tracking_lotf learning=shac seed=0 \
  output=results/tracking_lotf_new_s0
```

Navigation 有固定的 Depth、LiDAR 和分别记录的从零训练配方。新的 PPO 可选择
`navigation_depth_ppo_goal_observation` 或 `navigation_lidar_ppo_goal_observation`；
启用特权 Critic 的配方为 `navigation_*_scratch_ppo_asymmetric`。它们修改观测或
Critic 信息权限，不可直接与旧版结果视为同输入消融。

## 恢复与冻结评测

仅使用**本版本生成、结构和配置完全匹配的 v2 训练状态**恢复：

```bash
pixi run train experiment=tracking learning=apg seed=0 \
  output=results/tracking_apg_new_s0 \
  resume=results/tracking_apg_new_s0/checkpoints/latest.training.zip
```

训练与冻结归档有独立用途。完整训练状态保存 Actor、Critic、优化器、随机状态和
Environment；`checkpoints/step-*/policy.zip` 保存独立推理所需的 Actor 变量。
选择权重由 `run.json.selection.checkpoint` 引用。选模评测与独立 benchmark 的随机
种子分区不重叠；`episodes.csv` 记录全部失败分母，不能由总成功率反推单场景成绩。

独立测试命令及控制方法见[控制模型](control-models.md)；实验标准见
[验收要求](acceptance.md)。

## 历史证据

初始平台的三算法验收、初始化实验及从零训练调查发生在 v1 状态结构下。
历史指标、来源、失败与成本保存在 `results/acceptance/`、
[从零训练记录](scratch-training.md) 和 Git 历史。它们不是当前代码已经重新训练通过的证据。

本版本不接受旧 `.training.zip` 或 `.policy.zip`，也不提供旧权重转换与目录迁移命令。
旧权重和报告保留为只读研究证据；开始新的实验须重新训练并单独记录成本。
