# P5-01：几何场景与导航任务

Type: task
Status: blocked
Blocked by: P5-00
Engineering: planned
Experiment: not-started
Quality: not-evaluated
Owner: unassigned
Session: none
Run: none

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

实现圆柱森林、混合横杆/盒体、动态障碍及统一导航事件，完成静态和动态真实动作闭环与rscope回放。

## 验收

同源几何与时钟一致；固定种子复现；机体碰撞、0.5米到达、40秒超时、快速穿越与重置通过。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

待执行后填写；当前状态为计划。
