# P5-07：八单元正式训练

Type: task
Status: resolved
Blocked by: none
Engineering: passed
Experiment: completed
Quality: observed-low; threshold-unset
Owner: main
Session: p5-main
Run: experiments/p5-matrix-v2/queue-state.json

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

八个单元各训练8388608交互，保存完整过程、开发选模、独立开发和留出结果。

## 验收

每单元预算守恒；初始/中期/最终/最优身份明确；三档难度分别评测；失败和低分保留；留出不参与调参。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

用户已授权直接执行。八配方各 8388608、seed 0、7200 秒墙钟上限；冻结 PPO/D.VA 的共享传感器与网络。
`scripts/run_p5_learning_matrix.py` 串行调用同一组合入口，预算精确核验，开发选模后独立 dev32/heldout128 每难度。
旧失败目录保留，队列拒绝覆盖或自动暖启动；单元失败后继续其它独立单元并汇总失败。

2026-09-28 阶段核验：八训练单元全部完成，各 8388608 次，共 67108864 次交互。
学习组独立开发 768 回合、留出 3072 回合全部完成，冻结权重和开发选模身份通过汇总检查。
动态 MID360 PPO 留出 22/384，其余七单元 0/384；四 D.VA 单元全部越界，质量原因待诊断。
缺少的三组早期静态学习评测逐帧归档归入 P5-08；保存的原权重用于补采，新增训练交互为零。
证据：`experiments/p5-matrix-v2/queue-state.json`、
`docs/verification/p5-results-v2/training.csv`、`tmp/p5/handoff-integrity-20260928.log`。
