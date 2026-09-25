# 领域文档

本项目采用单一领域文档：`CONTEXT.md` 只记录项目术语，`docs/adr/` 记录影响长期维护的重要决策。
已接受的研究范围在 `.scratch/drone-platform/spec.md`，模块职责在 `docs/architecture.md`，
执行状态在任务单和 `docs/status.md`。这些文件各维护一种事实。

修改任务、模型用途、方法输入等概念时先读相关术语和决策。概念澄清后立即更新术语；
重要决策使用递增编号和简短理由。可逆的局部实现选择写入任务记录即可。

旧讨论迁入 `docs/research/alignment-history.md`，它提供历史证据；当前规格决定执行范围。
原始测试报告迁入 `docs/verification/`，历史测试只覆盖报告明确列出的条件。

来源：mattpocock/skills 的 `domain-modeling/SKILL.md`、`ADR-FORMAT.md`。
