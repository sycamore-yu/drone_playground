# 03：APG 与随机样条跟踪

Type: task
Status: resolved
Blocked by: 01
Engineering: passed
Experiment: completed
Quality: passed-development32-and-heldout128-for-three-runs
Owner: ChatGPT prime
Session: Chat On Steroids current conversation
Run: p2-figure8-apg-seed0-v1; p2-random-ppo-seed0-v1; p2-random-apg-seed0-v1

## 要交付

同一任务入口运行 Brax APG，并复用 LSY 随机样条生成器；研究者可以重放固定参考下 PPO/APG 的行为。

## 验收

- [x] 复用作者三次样条构造和任务时序，任务随机种子与策略随机种子独立并可重建。
- [x] 实际反传窗口包含动作经 Crazyflow 到任务损失的路径，相关局部有限差分核对。
- [x] 原生 APG 进行真实参数更新、完整预算训练、检查点保存和独立评测。
- [x] PPO 和 APG 使用同一版任务、模型、观测和测试清单；各自训练配置来源明确。
- [x] 八字和样条分别报告全时域完成、误差和失败，生成相同初态可比较的重放。

## 证据与交接

三次正式训练分别完成 655360、4194304、655360 次交互，检查点在运行内 best.json 索引。
三个检查点各自独立开发32/32、留出128/128完成；留出RMSE分别为 0.02574215、0.00991630、0.00761645米。
随机参考 bank 使用训练10000、开发20000、留出30000起的独立种子，哈希与实际回合种子核对通过。
随机 APG 最终实时显示发生目录切换错误，权重/预算/自身重放已完整；只恢复结果收尾，原错误和恢复校验保留。
三个实际运行和四组实验汇总见 `docs/verification/p2-independent-evaluation.json` 与 `p1-p2-results.json`。
APG 原生入口提供周期评估标量与初始/最终策略；PPO 提供初始/中期/最终策略记录。
