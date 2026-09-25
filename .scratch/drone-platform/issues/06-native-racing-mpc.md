# 06：原生竞速与优化控制

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

运行 LSY 原生赛道和 AttitudeMPC，连同已有采样控制方法提供完整执行、任务事件和可重放的评测结果。

## 验收

- [ ] 固定 LSY 版本并修复已确认的 Crazyflow 参数导入适配，保留上游差异与单位核验。
- [ ] 原生门序、过门方向和判定框保持版本一致；碰撞、完成、超时均有真实行为检查。
- [ ] acados 真正求解并执行返回控制，记录状态、预测、求解结果及决策延迟。
- [ ] 作者控制器完整运行所支持的任务，结果有逐回合完成/失败和运行身份。
- [ ] 采样 MPC 保留原任务与方法名称，移植到新任务时明确代价和约束改动。
- [ ] rscope 可重放完整轨迹与门进度；同一任务的后续学习方法沿用该协议。

## 证据与交接

填入作者/本地版本、适配差异、求解器和真实运行日志、原任务完整记录。
本阶段可与学习质量调试并行，依赖共同记录接口即可。
