# P5-07：八单元正式训练

Type: task
Status: blocked
Blocked by: P5-04,P5-05；协议与墙钟预算冻结；用户执行授权
Engineering: planned
Experiment: not-started
Quality: not-evaluated
Owner: unassigned
Session: none
Run: none

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

八个单元各训练8388608交互，保存完整过程、开发选模、独立开发和留出结果。

## 验收

每单元预算守恒；初始/中期/最终/最优身份明确；三档难度分别评测；失败和低分保留；留出不参与调参。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

待执行后填写；当前状态为计划。
