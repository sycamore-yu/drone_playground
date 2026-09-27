# P5-06：原生规划器闭环

Type: task
Status: blocked
Blocked by: P5-01,P5-02,P5-03
Engineering: planned
Experiment: not-started
Quality: not-evaluated
Owner: unassigned
Session: none
Run: none

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

连接独立ROS规划进程，完成EGO+D435、SUPER+MID360的静态/动态导航。

## 验收

保留原生建图与规划；深度输入/世界点云有数值证据；真实求解和跟踪；失效轨迹、时限、重置隔离已检查。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

待执行后填写；当前状态为计划。
