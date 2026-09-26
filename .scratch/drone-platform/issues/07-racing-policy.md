# 07：能够完成竞速的学习策略

Type: task
Status: resolved
Blocked by: 02,06
Engineering: passed
Experiment: completed
Quality: PPO-APG-passed-SHAC-valid-low-score
Owner: ChatGPT prime
Session: Chat On Steroids current conversation
Run: docs/verification/p4-learning-v1.json

## 要交付

在与作者控制器一致的赛道条件下分别训练 PPO、APG、SHAC，保存检查点，并与真实采样 MPC、AttitudeMPC 比较竞速表现。2026-09-26 用户明确要求三种学习方法全部进入本阶段。

## 验收

- [x] 方法获得的信息和动作类型明确，作者任务规则与控制器基线完全一致。
- [x] 训练预算、验证赛道、选择检查点规则及质量目标在运行前冻结。
- [x] 周期评估的固定轨迹显示真实过门顺序，TensorBoard 显示门进度、完成率和时间。
- [x] 新进程重载一个达到已确认 Level 0 目标的检查点，提供全部开发试次及失败轨迹。
- [ ] 高难度随机化在独立配置报告，明确低难度达标的适用范围。
- [x] PPO/APG/SHAC 各有完整预算、独立评测与回放；算法有效低分保留，工程错误单列。
- [x] 优化控制与学习策略在相同任务事件上评价；预设样条跟踪与自由规划的区别写入报告。

## 依赖说明

依赖 02 的训练入口和 06 的任务协议，不依赖八字策略已经达标。

## 证据与交接

PPO/APG的开发32和留出128均完赛；保存策略及冻结归一化的新进程重评完成。
SHAC完成655360次交互，开发集选中初始策略，留出0/128；最终训练权重及失败重放保留。
三种方法的独立回放均通过VS Code读取器逐帧核验，证据见`docs/verification/p4-replay-verification.json`。
本轮图形检查是MuJoCo保存状态渲染，窗口交互继续采用此前扩展交付的验证范围。
本交付仅为Level0预设轨迹跟踪；高难度随机化保持待测，后续单列配置及预算。
