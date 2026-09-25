# 03：APG 与随机样条跟踪

Type: task
Status: ready-for-agent
Blocked by: 01
Engineering: planned
Experiment: not-started
Quality: not-started
Owner: unassigned
Session: none
Run: none

## 要交付

同一任务入口运行 Brax APG，并复用 LSY 随机样条生成器；研究者可以重放固定参考下 PPO/APG 的行为。

## 验收

- [ ] 复用作者三次样条构造和任务时序，任务随机种子与策略随机种子独立并可重建。
- [ ] 实际反传窗口包含动作经 Crazyflow 到任务损失的路径，相关局部有限差分核对。
- [ ] 原生 APG 进行真实参数更新、完整预算训练、检查点保存和独立评测。
- [ ] PPO 和 APG 使用同一版任务、模型、观测和测试清单；各自训练配置来源明确。
- [ ] 八字和样条分别报告全时域完成、误差和失败，生成相同初态可比较的重放。

## 证据与交接

填入梯度检查、任务一致性证据、实际预算、固定参考校验值、策略和评测位置。
方法的有效低分保留为结果，质量提升另有明确假设和预算。
