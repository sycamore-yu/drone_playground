# P5-03：MID360与吞吐冻结

Type: task
Status: blocked
Blocked by: P5-01；总吞吐表另依赖P5-02
Engineering: planned
Experiment: not-started
Quality: not-evaluated
Owner: unassigned
Session: none
Run: none

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

复用MuJoCo-LiDAR建立点云观测与回放；测两种传感器在1/16/128环境的开销和训练内存。

## 验收

MID360扫描方向/相位/点时间明确；动态几何更新和批量隔离通过；按67108864交互推算墙钟预算并提交具体冻结表。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

待执行后填写；当前状态为计划。
