# ADR-0007：训练结果布局与后续 Orbax/TensorBoard 迁移

**Status:** Accepted · 原决定 2026-10-09；新布局修订 2026-10-10。

**Implementation:** 当前集成分支 `v0.2` 仍使用 v2 记录器；Orbax/TensorBoard v3
已在独立 Scratch 功能分支开发，本 ADR 不声称这些代码已经合入 `v0.2`。
正式实现应与对应代码、测试及 ADR-0014 一起集成并核对。

## 当前已经实现：v2

`simulation/records.py`、`learning/checkpoint.py`、`simulation/evaluation.py`
及 `tools/report_acceptance.py` 当前仍以 `run.json`、`metrics.jsonl`、
`checkpoints/latest.training.zip`、`checkpoints/step-*/policy.zip`、
`eval/<编号>/report.json` 与 `episodes.csv` 为运行合同。

该合同在早期验收时记录了 C5 连续选模、C6 实际更新/耗时、独立随机种子、
失败分母与配置来源。历史运行不重解释、不删除，也不把旧归档自动恢复为新状态。

## 已批准但尚未在本分支落地：v3

为了减少多级评测目录与重复模型导出，采用 Orbax Checkpoint 保存完整 Flax
训练 PyTree，用 TensorBoard 事件文件记录指标。每个 run 采用一份小型机器状态索引，
模型按照实际更新数编号；保留必要的最终评测来源与逐回合数据。

```text
results/<run_id>/
  config.yaml
  run.json
  events.out.tfevents.*
  checkpoints/
    model_000025/
      training/
      metadata.json
    model_000050/
      training/
      actor/                # 只有选中权重需要独立推理导出
      metadata.json
      screening/            # 仅在需要时存在
  episodes.csv
  report.json
  replays/
    best/                   # 按需生成
```

`run.json` 和最终 `report.json` **仍然需要保留**：前者记录可恢复状态、来源、
完成情况与选模索引，后者保存逐场景验收、种子和完整失败分母。
只用 TensorBoard 曲线无法独立复算这些机器判据。

训练状态必须能够恢复 Actor、Critic、优化器、随机键、Environment 与传感器历史。
仅在选中更新导出冻结 Actor，未选中中间权重按声明的保留策略清理。
回放只在明确请求时生成。原生 Planner/Controller 运行不创建空的网络 Checkpoint。

## 迁移原则与完成门禁

- 只有**新运行**启用 v3。旧 v2 训练记录与 ZIP、报告保持原样；不对活动运行改写格式。
- 同步替换所有真实读取方：训练、恢复、冻结推理、选模、验收、CLI 与脚本。
- 在对应功能分支完成从保存到恢复、选中权重推理、三种子判据与打包测试后再合入。
- 不因为结果目录迁移就声称 PPO/APG/SHAC 已经重新收敛。
- 该修订的详细实现设计以 Scratch 分支的 ADR-0014 为准；合并时统一 ADR 索引，
  不长期维护相互冲突的双份规格。

## 源码依据

- [当前记录器](../../src/drone_playground/simulation/records.py)
- [当前保存和恢复](../../src/drone_playground/learning/checkpoint.py)
- [当前训练入口](../../src/drone_playground/cli.py)
- [验收读取约定](../acceptance.md)
- [Orbax Checkpoint](https://orbax.readthedocs.io/en/latest/)
