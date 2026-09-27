# P5-04：感知PPO

Type: task
Status: blocked
Blocked by: P5-02,P5-03
Engineering: planned
Experiment: not-started
Quality: not-evaluated
Owner: unassigned
Session: none
Run: none

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

实现共享深度/点云编码器、策略/价值观测权限和两传感器PPO真实更新、独立加载。

## 验收

两种测量实际影响策略；部署输入没有障碍真值；奖励/终止正确；保存加载与恢复口径明确。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

待执行后填写；当前状态为计划。
