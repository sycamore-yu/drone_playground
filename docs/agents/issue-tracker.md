# 任务跟踪：本地 Markdown

新项目使用本地 Git，规格和任务放在 `.scratch/`。这是 Matt 的本地任务跟踪形式；
此目录存放持久设计与任务，临时调试文件存放在 `tmp/`。

- 功能目录：`.scratch/drone-platform/`。
- 规格：`spec.md`；阶段地图：`map.md`。
- 每个任务独占 `issues/NN-slug.md`，记录可验收能力、依赖、状态及证据。
- `Status: ready-for-agent` 表示可由智能体领取；依赖完成后方可执行。
- 执行中改为 `claimed`；验收后改为 `resolved`；阻塞使用 `blocked` 并写出具体条件。
- 状态解释和研究质量分别记录，任务单保留 `Engineering`、`Experiment`、`Quality`。
- 任务状态以任务单为准，`docs/status.md` 提供当前活动、阶段摘要与链接。
- 规格和任务均纳入 Git；外部 GitHub Issues/PR 的创建留待用户授权。

每次领取前检查既有 `Owner`、`Session`、`Run`；同一项工作继续使用原会话与运行标识。
一个主执行者维护共享接口，独立任务可在真实依赖满足后并行推进。

本项目对研究任务的补充：`Blocked by` 默认依赖该任务的 `Engineering: passed`，
所需具体输入在子任务中说明。`Experiment` 和 `Quality` 继续独立跟踪，因此一个算法仍在调参
不会阻断使用已交付接口的其他任务。最终完整交付同时核查预算结果和代表策略质量。

来源：mattpocock/skills，提交 `c55ee46073ed923f86ce59a5eb3b6d895095d1b7`，
`skills/engineering/setup-matt-pocock-skills/issue-tracker-local.md` 和 `to-tickets/SKILL.md`。
