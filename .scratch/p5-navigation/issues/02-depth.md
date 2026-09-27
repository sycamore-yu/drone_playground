# P5-02：D435完整观测链

Type: task
Status: blocked
Blocked by: P5-01
Engineering: planned
Experiment: not-started
Quality: not-evaluated
Owner: unassigned
Session: none
Run: none

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

完成场景到深度测量、观测历史、动作执行和回放的完整路径，验证原生单环境参照及批量实现。

## 验收

深度单位/光学坐标/视场/遮挡/无效值正确；单批量一致；动态更新、采样时间和环境隔离有证据。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

待执行后填写；当前状态为计划。
