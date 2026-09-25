# 04：SHAC 的完整跟踪训练

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

在同一 JAX/Brax 学习体系中补充 SHAC，研究者能看到短窗口训练、价值估计、实际控制与最终任务结果。

## 验收

- [ ] 算法对照 NVlabs/DiffRL 和 JAX 参考实现，记录来源、短窗口目标及必要移植差异。
- [ ] 终端价值的参数冻结和对状态的梯度分别核验，TD-λ、终止与截断自举有针对性测试。
- [ ] 实际策略/价值网络更新，梯度、价值损失与动作统计可观察；完成冻结预算。
- [ ] 保存策略、归一化和必要训练状态，独立评测与 rscope 轨迹可读。
- [ ] 将计算图正确、训练完整和任务质量分别报告，与 PPO/APG 同任务比较。

## 证据与交接

填入公式对应实现、相关测试、真实更新、训练曲线、检查点及开发评估。严禁用其他算法替代 SHAC 的方法身份。
