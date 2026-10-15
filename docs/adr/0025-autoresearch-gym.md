# ADR-0025: Autoresearch Gym 外部训练接入

状态：Accepted。批准依据：2026-10-15 本次用户明确要求创建分支并接入。
实现分支：`feat/autoresearch-gym`。尚未合入 `v0.2`。
记录与核对时间：2026-10-15T15:34:00Z。

## 决定

保留本项目的 CLI、Trainer、Simulation 和 Orbax。将 Autoresearch Gym 克隆到项目根目录下，
由一个外部适配器启动现有 CLI 并读取原生结果。两者使用独立 Python 环境。
上游固定版本和小补丁保存在本项目，避免依赖尚未发布的子模块提交。

自动候选比较需要固定预算。CLI 增加可选 `learning.max_interactions`，按完整更新计数，
预算到达时导出最后一次更新的 Actor。该模式跳过收敛选模，报告 `budget_exhausted`。
默认值为 `null`，保留现有训练到收敛或 Patience 结束的流程。

候选只能改变训练选项、Actor 配置和并行世界数。任务 JSON 固定物理实验、训练算法、
预算和评测条件。搜索使用 checkpoint-evaluation 种子段，最终正式 Benchmark 单独执行。

## 实现状态

`tools/autoresearch.py`、原生可选预算、Depth PPO 任务配置和回归测试已实现。
CLI 和适配器的 63 项相关测试通过。真实 GPU Depth PPO 从随机权重执行
16 次更新、8,192 次交互，保存 Orbax Actor 后完成六主场景共 12 回合冻结评测。
这轮短程训练得到 0 次成功、12 次碰撞，运行器正常退出并写入候选汇总。
此结果验证接入链路；策略收敛按现有正式流程另行评价。原始结果位于
`results/autoresearch/sessions/integration-check/`。
操作命令见 [Autoresearch Gym](../autoresearch.md)。
