# 实验产物

现役产物只分三类：

- `runs/`：真实执行记录。每个 run 不可变，路径为 `runs/<task>/<method>/<run_id>/`；保存解析配置、代码身份、依赖、状态、指标、checkpoint、评测报告和按需记录的回放。
- `selected/`：结果选择视图。只保存 run、report、checkpoint 的引用与摘要，不复制权重或回放。第一版入口为 [v1-18-cells.json](selected/v1-18-cells.json)。
- `scratch/`：预览、诊断、迁移和历史保留材料。它们不是正式 run，也不会因为目录名获得质量身份。

新运行默认使用时间戳加训练 seed 作为短 `run_id`；Task 和 Method 由父目录表达，不再重复写进名称。显式 `run_id` 仍可用于有意义的人工标签。

每个 run 的核心记录由 `RunRecorder` 自动生成，包括 `manifest.json`、`resolved-config.json`、`state.json`、`result.json`、`metrics/`、`checkpoints/` 和 `eval/`。完整 RScope/MuJoCo replay 默认关闭；确需保存时显式设置：

```bash
evaluation.record_replays=true
```

训练或评测本身仍保存数值报告、参数和选模记录；关闭 replay 避免为每次运行复制大型 `.mj_unroll`、`rscope_meta.pkl` 和模型资源。冻结策略 `play` 自动开启回放记录，Brax 训练显式开启 `training.publish_live` 时也会记录回放。

从已保留的历史证据更新第一版选择视图：

```bash
pixi run python scripts/tools/organize_experiments.py --import-legacy --apply
```

该命令只更新 `selected/`。省略 `--import-legacy` 时读取 `artifacts/verification/release-progress.json` 并要求所选现役 run 可访问。迁移前的 `main_result` 保存在 `scratch/legacy/main_result/`；其中 `original-run`、`rollouts` 和 `traces` 链接指向原冻结工作树，须保留这些工作树才能访问原回放。不要从目录名称推断“正式、最好或通过”，以 selection manifest、原始报告和 benchmark validation 为准。

`results/` 不进入源码发布包。删除或归档 run 前应先检查 `selected/` 是否仍引用它；`scratch/` 的清理可以更激进，但历史唯一证据和项目外备份仍应先核验。
