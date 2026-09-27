# P5-06：原生规划器闭环

Type: task
Status: in-progress
Blocked by: none
Engineering: verifying
Experiment: engineering-runs
Quality: not-evaluated
Owner: main
Session: p5-main
Run: p5-static-ego-v2-eng, p5-static-super-v1-eng

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

连接独立ROS规划进程，完成EGO+D435、SUPER+MID360的静态/动态导航。

## 验收

保留原生建图与规划；深度输入/世界点云有数值证据；真实求解和跟踪；失效轨迹、时限、重置隔离已检查。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

EGO 与 SUPER 原生运行目标编译通过。适配器保留上游建图/轨迹模块，不导入 ROS 到 Pixi。
EGO v2 已产出轨迹并驱动物理模型；发现上游手动目标接口硬编码高度，改用原生三维预设接口验证中。
SUPER 源文件有 LGPLv3-or-later 头部声明（仓库根目录没有 LICENSE）；继续采用外部进程集成并记录来源。
