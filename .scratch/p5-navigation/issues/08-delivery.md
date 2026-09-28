# P5-08：完整矩阵与交付

Type: task
Status: blocked
Blocked by: learning-archive-checkpoint-path
Engineering: archive-incomplete
Experiment: all-training-and-original-evaluations-completed
Quality: observed-low; threshold-unset
Owner: main
Session: p5-main
Run: experiments/p5-native-matrix-v2/queue-state.json

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

完成四个规划器单元评测，汇总36个难度格，提供可重建报告和同运行测量/轨迹/事件回放。

## 验收

4608个正式留出回合及1152个最终独立开发回合有逐回合记录；工程/实验/质量分列；原始证据与表一致。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

2026-09-28 当前状态：八训练与四规划器单元全部完成；开发 1152、留出 4608 回合齐全。
重新执行完整汇总退出 1，仅报告早期静态深度 PPO、深度 D.VA、MID360 PPO
开发/留出共 18 格缺全回合归档（1440 回合）。四份静态规划器补采已完成，场景隔离审计通过。
学习补采从 manifest.config 得到 checkpoint=null，在加载策略时 Path(None) 报 TypeError，
实际权重路径保存于原 eval/report.json.checkpoint；首个补采目录尚未创建。
修复入口：scripts/run_p5_archive_repairs.py:run_one；配置来源解释见
src/drone_playground/evaluation/execution.py:resolve_evaluation_config。
保存旧补采日志后，以原报告的冻结权重及摘要恢复补采；完成后重新汇总、绘图并验证
5760 条全回合归档及 72 个代表格。现有 replays.json 仅验证 30 格。
阶段核验日志：tmp/p5/handoff-integrity-20260928.log；下一步具体说明见 docs/status.md。

以下为历史实施记录：

v2 正式矩阵正在执行。主会话直接完成，不启动 dsh。
离线汇总 `scripts/summarize_p5.py --revision v2 --require-complete` 核验
8 个完整预算、dev 与 heldout 各 36 格、分母与场景/参数身份；未完成时退出非零。
报告在 `docs/verification/p5-results-v2/`，当前表保留缺失格，尚不能验收为完成。
全回合轨迹归档由 `evaluation/trace_archive.py` 保存；2 项读回/身份/终止/损坏检查通过。
早期缺归档运行以 `-archive-v1` 固定规则重评，保留原结果，不新增训练交互。
收尾必须验证全部 5760 个归档案例，代表回放另用 RScope 读取器验收。
任意案例 easy/31 已经读回 550 帧、位置误差 0；静态原始传感器第 150 步重建误差 0。
静态/动态 × 深度/LiDAR 四个原始输入重建组合均已核验，最大误差 0。
逐实例 train/dev/heldout 指纹隔离审计仍待执行，是汇总强制门槛。
