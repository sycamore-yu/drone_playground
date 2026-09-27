# P5-05：D.VA移植与点云扩展

Type: task
Status: blocked
Blocked by: P5-02,P5-03；复用P5-04网络接口
Engineering: planned
Experiment: not-started
Quality: not-evaluated
Owner: unassigned
Session: none
Run: none

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

对照官方D.VA移植到现有Brax/JAX，完成深度和点云两条真实更新链。

## 验收

固定观测代理目标的有限差分、动作导数、末端价值、终止和重置掩码通过；编码器有效更新；恢复已验证。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

待执行后填写；当前状态为计划。
