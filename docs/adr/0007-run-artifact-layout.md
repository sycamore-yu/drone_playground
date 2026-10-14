# ADR-0007：训练结果布局与后续 Orbax/TensorBoard 迁移

**Status:** Superseded by [ADR-0016](0016-results-layout-orbax-tensorboard.md) for new runs · 2026-10-10

**Implementation:** 本布局只描述 `layout_version=2` 历史产物；新 run 使用 Orbax/TensorBoard v3，原有 ZIP 结果保持只读或由已有进程按原格式续训，不批量迁移。

## 历史运行结果：v2

`simulation/records.py`、`learning/checkpoint.py`、`simulation/evaluation.py`
及 `tools/report_acceptance.py` 旧运行使用 `run.json`、`metrics.jsonl`、
`checkpoints/latest.training.zip`、`checkpoints/step-*/policy.zip`、
`eval/<编号>/report.json` 与 `episodes.csv` 为运行合同。

该合同在早期验收时记录了 C5 连续选模、C6 实际更新/耗时、独立随机种子、
失败分母与配置来源。历史运行不重解释、不删除，也不把旧归档自动恢复为新状态。

## 新运行结果：v3

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

- 新运行使用 v3。旧 v2 训练记录与 ZIP、报告保持原样；不对活动运行改写格式。
- 同步替换所有真实读取方：训练、恢复、冻结推理、选模、验收、CLI 与脚本。
- 在主分支核对保存/恢复、冻结推理、筛选和评测测试。
- 不因为结果目录迁移就声称 PPO/APG/SHAC 已经重新收敛。
- 详细合同见 Results v3 的 ADR，历史目录按来源保留。

## 源码依据

- [当前记录器](../../src/drone_playground/simulation/records.py)
- [当前保存和恢复](../../src/drone_playground/learning/checkpoint.py)
- [当前训练入口](../../src/drone_playground/cli.py)
- [验收读取约定](../acceptance.md)
- [Orbax Checkpoint](https://orbax.readthedocs.io/en/latest/)
