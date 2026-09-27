# P5-05：D.VA移植与点云扩展

Type: task
Status: in-progress
Blocked by: none
Engineering: passed; full-run selection resume verifying
Experiment: engineering-runs-completed
Quality: not-evaluated
Owner: main
Session: p5-main
Run: p5-static-depth-dva-seed0-v2-eng, p5-static-lidar-dva-seed0-v1-eng

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

对照官方D.VA移植到现有Brax/JAX，完成深度和点云两条真实更新链。

## 验收

固定观测代理目标的有限差分、动作导数、末端价值、终止和重置掩码通过；编码器有效更新；恢复已验证。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

用户已授权主会话直接完成 P5 剩余工作。接续未提交的 D.VA 实现；核对上游 actor detach、末端 bootstrap、学习率与完整恢复。修正超时后的 reset 状态误用及 critic 多次更新导致学习率过快衰减，补充真实梯度和恢复验证。

两个 GPU 工程运行各完成 262144 次交互，exit 0，actor/critic/encoder 实际更新。
37 项导航/D.VA/规划器接口测试通过；额外固定观测、完整动作—动力学—奖励—末端价值代理目标有限差分通过。
CPU 连续/恢复完整状态比较通过；GPU 跨进程续训及开发选模继承另行验证中。
早期深度 v1 梯度 NaN 的失败保留：盒体内部范数零点选择有限次梯度后 v2 通过，前向距离和碰撞规则未变。
