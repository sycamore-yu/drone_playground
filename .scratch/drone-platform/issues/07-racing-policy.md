# 07：能够完成竞速的学习策略

Type: task
Status: ready-for-agent
Blocked by: 02,06
Engineering: planned
Experiment: not-started
Quality: not-started
Owner: unassigned
Session: none
Run: none

## 要交付

在与作者控制器一致的赛道条件下训练 PPO，保存能完成门序的代表策略，并比较真实竞速表现。

## 验收

- [ ] 方法获得的信息和动作类型明确，作者任务规则与控制器基线完全一致。
- [ ] 训练预算、验证赛道、选择检查点规则及质量目标在运行前冻结。
- [ ] 周期评估的固定轨迹显示真实过门顺序，TensorBoard 显示门进度、完成率和时间。
- [ ] 新进程重载一个达到已确认 Level 0 目标的检查点，提供全部开发试次及失败轨迹。
- [ ] 高难度随机化在独立配置报告，明确低难度达标的适用范围。

## 依赖说明

依赖 02 的训练入口和 06 的任务协议，不依赖八字策略已经达标。

## 证据与交接

填入检查点、完整门序事件、各试次结果、重评命令和控制器对照。
