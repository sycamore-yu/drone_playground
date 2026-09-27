# P5-04：感知PPO

Type: task
Status: resolved
Blocked by: P5-02,P5-03
Engineering: passed
Experiment: engineering-run
Quality: separate (P5-07)
Owner: main-session
Session: p5-main
Run: p5-static-depth-ppo-seed0-v2-eng

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

实现共享深度/点云编码器、策略/价值观测权限和两传感器PPO真实更新、独立加载。

## 验收

两种测量实际影响策略；部署输入没有障碍真值；奖励/终止正确；保存加载与恢复口径明确。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

实现：`learning/perception.py` 是 encoder 与 actor/critic 契约的唯一所有者。
`SensorLayout` 把 2420 维观测拆回本体 20 维与 `(history, points, channels)` 传感块；
`SharedEncoder` 对深度用卷积、对点云用逐点 MLP + 最大/均值汇聚，两者输出同宽 embedding
（128×4 帧）；`PerceptionActor` / `PerceptionCritic` 在共享 embedding 之上接各自的头。
PPO 的 `NormalDistribution` 消费 `(loc, scale)` 二元组，`NormalTanhDistribution` 消费拼接
向量，头形状按分布实际契约生成而不是假设。

**信息边界**：actor 与 critic 消费**完全相同**的张量；`privileged_critic_fields()` 返回
空的特权字段列表，并显式列出被拒绝的候选（障碍中心/尺寸、障碍速度、scenario_id/难度、
逐障碍净空、最近障碍距离）。`tests/test_perception_ppo.py` 断言
`privileged_fields == []`、actor 与 critic 共享观测、场景 manif est 与未来动态不可见，
并验证给定相同观测时策略输出逐元素相同。

**验收检查** `tests/test_perception_ppo.py` 共 7 项通过：布局与环境观测维度一致并可序列化、
split 还原声明的帧形状、两种传感器给出**完全相同的 actor/critic 头形状**、单条与批量观测
都可用、策略无法区分观测之外的场景、观测无场景真值通道、checkpoint 保存/重载逐元素复现
动作、以及真实 PPO 更新确实移动了卷积 encoder 的参数（非仅头部）。

**真实运行**（工程预算，非正式预算）：
`experiments/p5-static-depth-ppo-seed0-v2-eng` 262144 交互、`training/sps=3328.0`、
act or 参数位移 11.86、退出码 0；`experiments/p5-static-lidar-ppo-seed0-v1-eng`
同为 262144 交互、退出码 0。两次都写出 3 个检查点、3 份开发集评测与逐难度回放。
低分如实保留：深度单元最终成功率 0、碰撞率 0.167，属质量未达标而非工程失败。

修正过程中发现并解决的两个真实缺陷：Brax 的 value network 必须 squeeze 尾轴，
否则 PPO 损失把 `(T,B,1)` 与 `(T,B)` 广播失败；LiDAR 窗口表不能缓存在 jit 作用域内，
否则 tracer 会逃逸（`UnexpectedTracerError`）。
