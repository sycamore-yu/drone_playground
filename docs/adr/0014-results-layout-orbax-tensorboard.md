# ADR-0014：精简 Results 目录（Orbax + TensorBoard）

**Status:** Accepted · 2026-10-10

**Supersedes:** [ADR-0007](0007-run-artifact-layout.md) 的新 run 布局。

## Context

旧版每个训练 run 保存 `metrics.jsonl`、`latest.training.zip`、
`checkpoints/step-*/policy.zip` 和 `eval/001/` 等阶段目录，Hydra
还写入 `.hydra/` 与 `cli.log`。这使反复训练容易积累大量临时文件。
MuJoCo Playground 使用每实验日志目录与 TensorBoard 可选监控，
Isaac Lab 分离日志与权重；当前 Flax/JAX 推荐 Orbax 保存 PyTree 状态。

## Decision

新 run 的正式目录如下：

```text
results/<run_id>/
├── config.yaml
├── events.out.tfevents.*
├── run.json                      # 小型机器状态/来源/完成性索引
├── checkpoints/
│   ├── model_000025/
│   │   ├── training/             # Orbax 完整训练状态
│   │   └── metadata.json
│   ├── model_000050/
│   └── model_000075/
│       ├── training/
│       ├── actor/                # 仅选模成功导出
│       ├── metadata.json
│       ├── screening/            # 可选轻量评测数据
│       └── report.json           # 可选完整候选评测
├── episodes.csv                 # 正式冻结评测生成
├── report.json                  # 主场景、扩展场景成绩及分母
└── replays/
    └── best/                    # 正式评测开启回放后按需生成
```

`run.json` 和正式 `report.json` 是原草图之外保留的**必要数据索引**：
前者用于断点恢复、失败识别和三种子矩阵汇总，后者保存逐场景的真实成绩。
新 run 不再生成 `metrics.jsonl`、`latest.training.zip`、独立 `eval/`
或 Hydra `.hydra/`/自动日志。临时 stdout 和调参提案归矩阵根目录的
`.scheduler/`，不放入单个 run。

Orbax 保存完整 Flax `TrainingState`，含 Actor/Critic、优化器、随机键、
物理/传感器与 GRU 历史。每个更新对应一个 `model_NNNNNN`；
选模通过时在该目录导出独立推理 Actor，不复制另一份权重目录。
默认只保留最近三个仍可续训的模型目录（`results.keep_checkpoints=3`），
另永久保留任何已经选中的 `actor/` 所属目录，避免感知传感器状态快照无限增长。
TensorBoardX 使用训练更新次数作为标量 step，机器状态索引不再复写整条指标流。
`replays/best` 只按需生成：全帧点云/深度回放可能比训练指标本身更昂贵。

## Migration

**仅新 run 使用 v3/Orbax/TensorBoard，不改写正在执行的旧 run。**
旧 run 保留 `layout_version=2`、ZIP 训练状态、JSONL 记录与原选模合同。
续训按原版本加载，矩阵可以混合读取 v2/v3；不移动已验收权重。

## Evidence

Orbax 0.12.7 已完成 Flax PyTree + typed PRNG 的实际保存/恢复试验，
TensorBoardX 2.6.4 已生成真实 TensorBoard 事件文件。文件目录与选模入口
由对应测试覆盖；不将格式迁移等同于三算法 18 单元已重新收敛。
