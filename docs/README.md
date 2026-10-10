# 文档入口

[功能规格](spec.md)定义当前唯一的 **S1–S4 / L1–L4 八项验收**；[验证记录](validation.md)和[历史验收说明](acceptance.md)描述已有执行证据。旧 C/S/L 二十项记录只作追溯，不作为另一套任务。术语使用根目录的 [GLOSSARY.md](../GLOSSARY.md)。

## 架构决策

`Accepted` 表示用户已明确批准设计，提问或要求评估不构成批准。`Withdrawn` 表示提案已撤回，不作为实施依据。每份 ADR 的实现说明区分当前能力和待实现改动，不能用 ADR 代替运行证据。

| 决策 | 内容 |
|---|---|
| [0001](adr/0001-simulation-learning-boundary.md) | Simulation/Learning 分工；2026-10-09 补充构造阶段的方法组合和两种执行方式 |
| [0002](adr/0002-crazyflow-upstream-and-style.md) | 官方 Crazyflow、依赖及代码规范 |
| [0003](adr/0003-project-scope.md) | 首发研究范围和验收 |
| [0004](adr/0004-native-method-integration.md) | 原生 C++/ROS 接入 |
| [0006](adr/0006-native-crazyflow-randomization.md) | 使用原生 reset/step pipeline 随机化物理 |
| [0007](adr/0007-run-artifact-layout.md) | 权重与选模成绩同目录；统一独立评测；按需创建轨迹和回放 |
| [0008](adr/0008-independent-backward-dynamics-model.md) | 独立选择 `learning.backward_model`；模型数学实现与训练求导连接分工 |
| [0009](adr/0009-delayed-data-in-episode-state.md) | 命令和测量延迟缓冲归属对应环境实例，支持局部 reset 与恢复 |
| [0010](adr/0010-training-only-privileged-ppo-critic.md) | 训练专用特权 PPO Critic；Actor 与冻结评测使用原设备观测 |
| [0011](adr/0011-independent-ppo-goal-observation.md) | 独立 PPO 对照增加目标高度与距离；原验收批次保留十维观测 |
| [0012](adr/0012-rl-environment-and-method-interface.md) | Brax 环境与状态合同、网络／Method／动作解耦，Brax PPO/APG 作为可选对照 |
| [0013](adr/0013-three-stage-learning-acceptance.md) | 轻量筛查→完整候选选模→独立冻结 Benchmark；旧 run 保留原协议；吞吐测量 |
| [0014](adr/0014-results-layout-orbax-tensorboard.md) | 新 Results v3：Orbax + TensorBoard；旧 ZIP 训练保留原位置 |
| [0015](adr/0015-eight-acceptance-gates.md) | C/L/S 20 项去重为 S1–S4 与 L1–L4 八项；原质量门槛保留 |
| [0016](adr/0016-training-patience-and-frozen-selection.md) | 800 updates 后 6 次轻量筛查无改善即结束配方；首次完整通过即独立冻结一次 |
| [0017](adr/0017-procedural-training-fixed-navigation-benchmark.md) | Navigation 训练采用每环境程序化障碍场景；正式 S01–D06 几何冻结，只用于独立评测 |

当前实验分支的新训练使用 Orbax/TensorBoard v3；已有 v2 ZIP 训练作为只读
历史证据保留，调度器不得自动恢复旧三连续训。配置、控制器、延迟、随机化及反向模型见
[控制模型说明](control-models.md)。历史结果属于独立研究证据，不能直接视为
当前训练的收敛证明。

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

## 阶段性实验记录

版本控制只保留稳定的规格、ADR、可复现入口和正式研究结论。
每轮参数尝试不新增 Markdown：失败模型、过程指标、日志、独立评测及来源
统一归入被 Git 忽略的 `results/scratch_v02/`，由单一 `matrix.json` 汇总。
临时诊断脚本与缓存放 `tmp/`；历史实验文档从当前索引退出，由 Git 历史保留。
最终有独立冻结证据的结论才回填 README，且不会将历史权重的成绩归入从零收敛。

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
