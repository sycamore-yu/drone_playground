# P5-01：几何场景与导航任务

Type: task
Status: resolved
Blocked by: P5-00
Engineering: passed
Experiment: engineering-run
Quality: separate (P5-07)
Owner: main-session
Session: p5-main
Run: p5-nav-smoke-static-v3

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

实现圆柱森林、混合横杆/盒体、动态障碍及统一导航事件，完成静态和动态真实动作闭环与rscope回放。

## 验收

同源几何与时钟一致；固定种子复现；机体碰撞、0.5米到达、40秒超时、快速穿越与重置通过。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

实现：
`src/drone_playground/tasks/scenes/navigation.py`（SANDO 风格解析几何与运动、密度驱动生成、
起终点连通占据搜索）、`src/drone_playground/tasks/navigation.py`（统一导航任务、
逐物理子步碰撞、到达/碰撞/越界/数值失效事件）、
`src/drone_playground/evaluation/navigation.py`、`src/drone_playground/runs/navigation_scene.py`
（逐实例 MuJoCo 回放模型）、`configs/scene|task|observation|objective|policy|experiment/p5_navigation_*.yaml`。

真实闭环运行 `experiments/p5-nav-smoke-static-v3`：PPO 实际更新 32768 交互，
开发集 96 回合（三档难度各 32）逐回合记录，每档导出 4 份 rscope 回放；退出码 0。
随机初始策略 96/96 越界，作为真实低分保留。命令见运行目录 `command.txt`。

验收检查 `tests/test_navigation.py` 共 17 项全部通过，覆盖任务书要求的每一项：
同源几何与时钟（设备端运动函数对照宿主解析式、场景时间 == 物理步计数 × dt）、
固定种子复现与 train/dev/heldout 几何隔离、三档难度密度落在 SANDO 目标 ±0.03、
机体碰撞（圆柱侧面/顶面、盒体边界、非激活槽位）、0.5 m 到达、碰撞同一步优先、
越界与数值失效区分、40 秒/2000 步超时（阻尼高度保持真实飞满全程并判超时）、
快速穿越（40 m/s 在一个控制步内穿过 0.2 m 薄杆仍被捕获）与重置清空。

冻结的工程参数与来源偏差见
[P5来源清单](../../../docs/verification/p5-source-inventory.json)
的 `frozen_engineering_parameters` 与 `p5_01_measurements`；其中记录了两个实测事实：
Crazyflow 四元数为 xyzw（默认姿态 [0,0,0,1]），以及 first_principles/cf2x_L250 的
配平推力约在归一化 0.40（0.3514 N = 1.12·m·g），此前记录的 m·g 配平会掉高。

2026-09-27 P5-05—08 自审补充：地面与障碍物统一纳入最近净空/球体碰撞，
修复参考点尚在 z>0 时机体已触地却未失败的遗漏。上边界和侧边界仍为飞行范围，
没有虚构物理墙面。回归同时验证触地优先、仍单独记录越界，以及安全离地不误报。
旧正式 v1 队列中止、原始文件保留，新正式矩阵全部从头使用 v2；不混用两套协议。
