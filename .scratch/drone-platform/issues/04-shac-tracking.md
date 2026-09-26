# 04：SHAC 的完整跟踪训练

Type: task
Status: resolved
Blocked by: 01
Engineering: passed
Experiment: completed
Quality: mixed-valid-results
Owner: current-chat
Session: Chat On Steroids current conversation
Run: docs/verification/p3-p4-results.json

## 要交付

在同一 JAX/Brax 学习体系中补充 SHAC，研究者能看到短窗口训练、价值估计、实际控制与最终任务结果。

## 验收

- [x] 算法对照 NVlabs/DiffRL 和 JAX 参考实现，记录来源、短窗口目标及必要移植差异。
- [x] 终端价值的参数冻结和对状态的梯度分别核验，TD-λ、终止与截断自举有针对性测试。
- [x] 实际策略/价值网络更新，梯度、价值损失与动作统计可观察；完成冻结预算。
- [x] 保存策略、归一化和必要训练状态，独立评测与 rscope 轨迹可读。
- [x] 将计算图正确、训练完整和任务质量分别报告，与 PPO/APG 同任务比较。

## 证据与交接

实现为`learning/shac.py`，接口/梯度/TD-λ/续训证据在`tests/test_shac.py`；
8个跟踪单元均完成655360次交互，6个达到当前质量门槛。两项随机跟踪低分保留初始及最终训练权重。
结果与逐项文件摘要见`docs/verification/p3-p4-results.json`，原生回放逐帧检查见`p3-replay-verification.json`。
