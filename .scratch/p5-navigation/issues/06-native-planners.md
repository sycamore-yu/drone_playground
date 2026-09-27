# P5-06：原生规划器闭环

Type: task
Status: resolved
Blocked by: none
Engineering: passed
Experiment: engineering-runs-completed
Quality: not-evaluated
Owner: main
Session: p5-main
Run: p5-static-ego-v3-eng, p5-static-super-v3-eng, p5-dynamic-ego-v2-eng, p5-dynamic-super-v1-eng

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
EGO 使用原生三维预设接口解决手动目标硬编码高度问题。静态 3/3 到达，28 次轨迹发布；
动态 2/3 到达，40 次发布，hard 的 1 次碰撞保留。
SUPER 静态 3/3 到达、463 次轨迹发布；动态 6/6 到达、729 次发布。
双工作进程隔离工程运行两种规划器均 6/6 到达，分别 67/807 次真实轨迹发布。
失效时间戳/非有限轨迹与控制契约测试通过，桥按 SHA 固定，实际输入样本、启动参数和日志随回合保存。
9 项原生接口/回放身份测试通过；构建身份见 `docs/verification/p5-native-build.json`。
这些是工程检查，不作为完整留出统计；四单元正式矩阵归 P5-08。
SUPER 源文件有 LGPLv3-or-later 头部声明（仓库根目录没有 LICENSE）；继续采用外部进程集成并记录来源。
