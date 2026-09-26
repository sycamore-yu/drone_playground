# 06：原生竞速与优化控制

Type: task
Status: resolved
Blocked by: 01
Engineering: passed
Experiment: completed
Quality: both-methods-passed-heldout128-threshold
Owner: current-chat
Session: Chat On Steroids current conversation
Run: docs/verification/p4-optimization-v2.json

## 要交付

运行 LSY 原生赛道和 AttitudeMPC，连同已有采样控制方法提供完整执行、任务事件和可重放的评测结果。

## 验收

- [x] 固定 LSY 版本并修复已确认的 Crazyflow 参数导入适配，保留上游差异与单位核验。
- [x] 原生门序、过门方向和判定框保持版本一致；碰撞、完成、超时均有真实行为检查。
- [x] acados 真正求解并执行返回控制，记录状态、预测、求解结果及决策延迟。
- [x] 作者控制器完整运行所支持的任务，结果有逐回合完成/失败和运行身份。
- [x] 采样 MPC 保留原任务与方法名称，移植到新任务时明确代价和约束改动。
- [x] rscope 可重放完整轨迹与门进度；同一任务的后续学习方法沿用该协议。

## 证据与交接

LSY提交、acados0.5.1与兼容补丁保存在源码/运行清单。正式原生扰动v2：AttitudeMPC117/128完赛，
11碰撞；采样MPC128/128完赛。每方法4×32独立种子分片，合并时重验种子和原报告摘要。
首轮无扰动v1为已保留诊断。结果见`p4-optimization-v2.json`；逐步求解日志在分片`eval/solver-steps.json`。
完整轨迹及便携模型由已交付VS Code读取器验证，见`p4-replay-verification.json`；
保存状态的MuJoCo图形检查见`p4-visual-check.json`。
