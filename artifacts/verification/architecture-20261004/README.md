# 2026-10-04 架构重构验收

基线为 `69b1c392dcd52fdf7d85025797141b397e100010` 加当时已有的工作区修改。
本次完成用户批准的构造、公共执行、配置、数值、学习、存储、回放、ROS 和命名迁移。
没有提交、推送、重建 ROS 容器或覆盖历史结果。

## 实现

环境共享资源由 `environments/initialization.py` 创建和释放；任务保留自己的初态、参考和事件规则。
`env.freq` 与实际物理时钟决定子步数。`ActionTransition` 共用一个 JAX 子步循环，处理动作到达、
碰撞累计和可选的首次终止冻结；任务不再提供私有 `advance_checked` 物理循环。
原生 Racing 周期末判定、PointMass 论文离散映射及训练梯度保持原语义。

`configuration.py` 和 `app.py` 分别负责配置准备与执行，旧 `composition.py` 已删除。
数学公式归入 `numerics.py`；任务奖励与学习损失分开。持久化模块不构造网络或环境，
冻结推理位于 `learning/inference.py`，三个循环训练入口共享快照与更新编排。
RScope 发布、ROS launch、离线场景检查和导航回放已移到对应职责模块。

## 验证证据

| 检查 | 结果 | 记录 |
|---|---|---|
| 配置、存储、运行边界 | 131 项通过 | `logs/boundaries-final.xml` |
| 训练、恢复、公共入口与资源释放 | 40 项通过，另有 3 个子测试 | `logs/training.xml` |
| 回放、事件、PointMass 与重置 | 首次 28 项通过，1 项旧测试夹具失败；该项随后修复并通过 | `logs/replay-events.xml`、`logs/final-targets.xml` |
| 最终定向检查 | 50 项通过 | `logs/final-targets.xml` |
| 合并后的独立用例 | 248 项最新结果均通过 | `test-results.json` |
| 测试发现 | 579 项可收集，无导入错误；未执行全套 579 项 | `logs/collection-final.log` |
| 数值对照 | 九类环境、388 个数组在原定容差内一致 | `logs/transitions-final.log`、`logs/acceleration-final.log` |
| GPU 真训练 | BPTT 4 个环境步，参数变化 L2 = 0.09126696013548118 | `gpu-result.json`、`logs/gpu-smoke.log` |
| 原生 ROS 短闭环 | SUPER 执行 17 次、EGO 执行 46 次原生控制 | `native-closed-loop.json`、`native-ego-startup.json` |
| 安装包 | wheel 构建、隔离安装、仓库外 JIT reset/step 通过 | `installed-verification.json`、`logs/wheel.log` |
| 保护检查 | 16,124 个原文件摘要一致，无删除 | `protected-verification.json` |
| 静态检查 | Ruff F/I 通过，169 个改动 Python 文件格式检查通过 | `logs/lint-core.log`、`logs/format-final.log` |

独立用例按 `(classname, name)` 合并，后续结果覆盖同一用例的较早结果；不累加重复测试。
初始化失败释放、角速度控制推力记录及旧入口修复均保留了失败复现日志。
数值比较使用 `rtol=atol=3e-6`，不是所有数组逐位相等的声明。

全规则 Ruff 仍报告 565 项，主要是缺少文档字符串及其他规范项，详见
`logs/lint-final-statistics.log`。本次没有为消除全部规则报告而扩大到整仓风格重写。
上述结果证明本轮工程接入，不证明策略达到正式任务质量门槛；GPU 探针只有一次参数更新。

机器汇总在 `verification.json`，变更路径和摘要在 `changed-files.json`。
原始快照、未删节探针与运行目录位于 `tmp/architecture-implementation-20261004/`。
当前架构见 [架构文档](../../../docs/architecture.md)，实际调用方法见
[操作手册](../../../docs/runbook.md)。
