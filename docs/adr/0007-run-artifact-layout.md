# ADR-0007：权重与选模成绩同目录，独立评测统一归档

**Status:** Accepted · 2026-10-09

**Implementation:** 新运行记录、CLI 和验收读取代码已支持本布局。当前程序只处理 v2 产物；历史结果保持只读，不提供兼容性迁移。

## Context

当前运行把同一步的权重和选模成绩分别放在 `checkpoints/`、`checkpoint_eval/`。`events/` 只有一个日志文件，根目录 `rollouts/` 被无条件创建但没有产物。最终评测按场景建立多级目录，使一次运行难以浏览。

权重、续训状态、选模成绩和独立评测承担不同职责。可以减少目录与重复记录，同时保留这些实验语义。

## Decision

采用以下目标布局。`step-*` 表示训练更新编号；`eval/001` 表示一次独立评测。

```text
results/<run_id>/
  config.yaml
  run.json
  metrics.jsonl
  checkpoints/
    latest.training.zip
    step-000025/
      policy.zip
      report.json
      episodes.csv
    step-000050/
      policy.zip
      report.json
      episodes.csv
    step-000075/
      policy.zip
      report.json
      episodes.csv
  eval/
    001/
      report.json
      episodes.csv
      trajectories.npz
      replays/
        S01-0000/
        D03-0004/
```

`checkpoints/step-*/policy.zip` 与评价该权重的选模结果放在一起，取消独立的 `checkpoint_eval/` 目录。没有发生评测的保存点仅有权重，不创建空报告。

`latest.training.zip` 用于完整续训，保存优化器、环境和随机状态；`policy.zip` 用于独立推理。两者保留不同用途，不合并成一种归档。

取消单文件 `events/` 目录，把追加日志放在根目录 `metrics.jsonl`。保留训练指标、恢复、场景切换和评测事件。

独立评测统一写入 `eval/<编号>/`，包括最终 benchmark、原生方法和鲁棒性评测。报告记录用途、实际配置、种子以及权重或方法身份。重复评测使用新编号，不覆盖旧结果。目录改名不取消最终 benchmark 协议。

一次评测的八个场景写入同一个 `episodes.csv`，使用 `scene` 列区分；`report.json` 给出逐场景统计。回合身份包含场景、初始化种子和世界索引。不能把不同场景的失败分母或验收门槛合并掉。

完整数值轨迹和回放按需保存；`trajectories.npz`、`replays/` 仅在实际产物生成时创建。轨迹必须能映射回逐回合表。选模评测默认保留统计及逐回合记录，完整轨迹和回放由实验配置开启。

根目录不创建空 `rollouts/`。回放归属于产生它的评测。选定权重由 `run.json` 引用原始权重文件，取消重复的 `selection.json`，不另复制 `best.policy.zip`。

## Consequences

训练选模与独立最终评测仍使用分开的种子和选择用途。C5 的连续通过历史、C6 的成本记录、初始化来源及全部失败回合继续保留。

2026-10-09 的收尾决定：旧结果布局不再作为当前代码的输入，也不继续维护
`migrate_results.py`、`selection.json` 或 `checkpoint_eval/update-*` 的兼容读取。
历史报告、逐回合数据和冻结归档保留原始字节用于追溯，但本版本不加载 v1 模型。
恢复与评测必须使用同结构、同动作合同的 v2 归档；不以搬动目录伪造兼容性。

## Evidence

- [当前记录器](../../src/drone_playground/simulation/records.py)：`RunRecord`。
- [当前训练保存入口](../../src/drone_playground/cli.py)：`train`；[评测入口](../../src/drone_playground/simulation/evaluation.py)：`evaluate`。
- [验收读取约定](../acceptance.md)。
- 2026-10-09 检查 `results/acceptance_depth_initialized_apg_s1`：三个选模更新、八个最终评测场景、一个事件日志、空的根目录 `rollouts/`。
