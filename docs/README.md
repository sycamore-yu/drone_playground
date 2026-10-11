# 文档入口

[功能规格](spec.md)定义研究范围；[验证记录](validation.md)和[验收说明](acceptance.md)描述已有执行证据。术语使用根目录的 [GLOSSARY.md](../GLOSSARY.md)。

**集成文档核对：2026-10-10T01:27:02Z（UTC）。** 当前集成分支为 `v0.2`。功能分支的 ADR 和文档仅是各自的 Git 快照；合并时要核对现有决策、编号和索引，不能以分支副本覆盖新决定。协作规则见 [AGENTS.md](../AGENTS.md)。

## 架构决策

`Accepted` 表示用户已明确批准设计，提问或要求评估不构成批准。`Withdrawn` 表示提案已撤回，不作为实施依据。每份 ADR 的实现说明区分当前能力和待实现改动，不能用 ADR 代替运行证据。

| 决策 | 内容 |
|---|---|
| [0001](adr/0001-simulation-learning-boundary.md) | Simulation/Learning 分工；2026-10-09 补充构造阶段的方法组合和两种执行方式 |
| [0002](adr/0002-crazyflow-upstream-and-style.md) | 官方 Crazyflow、依赖及代码规范 |
| [0003](adr/0003-project-scope.md) | 首发研究范围和验收 |
| [0004](adr/0004-native-method-integration.md) | 原生 C++/ROS 接入 |
| [0006](adr/0006-native-crazyflow-randomization.md) | 使用原生 reset/step pipeline 随机化物理 |
| [0007](adr/0007-run-artifact-layout.md) | 当前 v2 结果约定；已批准后续 Orbax/TensorBoard v3，尚未并入本分支 |
| [0008](adr/0008-independent-backward-dynamics-model.md) | 独立选择 `learning.backward_model`；模型数学实现与训练求导连接分工 |
| [0009](adr/0009-delayed-data-in-episode-state.md) | 命令和测量延迟缓冲归属对应环境实例，支持局部 reset 与恢复 |
| [0010](adr/0010-training-only-privileged-ppo-critic.md) | 训练专用特权 PPO Critic；Actor 与冻结评测使用原设备观测 |
| [0011](adr/0011-independent-ppo-goal-observation.md) | 独立 PPO 对照增加目标高度与距离；原验收批次保留十维观测 |

当前集成分支 `v0.2` 的代码和训练产物使用 v2-only 合同。配置、控制器、延迟、随机化及
反向模型见[控制模型说明](control-models.md)。历史结果与当时验证属于独立的研究
证据，不再由当前加载器自动兼容。最近验证状态见[验证记录](validation.md)。

参考模型索引和历史评测报告见[研究资产](../research/checkpoints/README.md)。
`research/checkpoints/reports/` 属于有来源的历史证据，不能把它和运行时缓存一起清除。

2026-10-09 的数据清理移除了 `results/` 下 **75 个 v1 完整训练归档**（2.19 GiB）；
**546 个冻结策略归档**、历史验收报告、轨迹与回放仍然保留。这些 v1 冻结策略只作
来源审计，不能直接交给当前 v2 加载器。逐文件删除清单和 SHA-256 保存在本地忽略的
`tmp/neat-freak/legacy-training-deleted-20261009.json`。不从该清单推导可恢复的训练参数。

## 设计原则

[ADR-0001](adr/0001-simulation-learning-boundary.md) 记录 Simulation/Learning 的职责与
方法组合，数学模型与 Controller 可以独立调用。Trajectory 保持宿主 NumPy 表示，
SE3 暂不加入。原迭代审查保留在 Git 历史，不作为另一套现役规格。

## 运行说明

[训练接口](training.md)、[实验配方](experiments.md)、[原生方法](ros_planner.md)、[回放](replay.md)。

## 代码索引与 Wiki

本机使用 GitNexus 1.6.10。GitNexus 的索引和生成的 Wiki 均在被 Git 忽略的
`.gitnexus/`，不是另一套权威手写文档。当前 Wiki 使用 CPA 的现有
`http://127.0.0.1:8317/v1` 接口及既有凭据，模型为
`deepseek/deepseek-v4.1-flash`。本机的私有启动脚本不会复制密钥到仓库。
本轮 GitNexus 1.6.10 已成功生成 **24 页** Wiki。每次代码或文档提交后使用下面的
命令增量刷新，并核对 `.gitnexus/meta.json.lastCommit`、
`.gitnexus/wiki/meta.json.fromCommit` 与 `git rev-parse HEAD` 一致；不在此处手写提交号。

```bash
export PATH="$HOME/.nvm/versions/node/v24.20.0/bin:$PATH"
gitnexus analyze --index-only --name new_drone_playground
~/.local/bin/gitnexus-wiki-cpa --force .
```

生成后查看 `.gitnexus/wiki/index.html`；生成式 Wiki 用于导航，事实以源码、
ADR、实测运行报告及手写文档为准。
