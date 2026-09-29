# Native planners

This directory owns the ROS1 runtime used by the native navigation planners in
`drone_playground`. It is independent of FlightBench and builds from
`ros:noetic-ros-base-focal`.

`versions.env` pins the upstream planner repositories and commits. `setup.sh`
checks out those revisions under the ignored `sources/` directory, builds the
project-owned `drone-playground-ros1:noetic` image, starts the
`drone-playground-ros1` container, and builds EGO-Planner and SUPER in isolated
catkin workspaces under `/opt/drone_playground/planners`.

Run:

```bash
bash native_planners/setup.sh
```

The simulator stays outside ROS. Each evaluation episode starts its own ROS
master inside this container and communicates with the bridge using JSON lines
over `docker exec` standard input/output. The planner source checkouts and build
context are intentionally excluded from Git; upstream license files remain in
their respective source trees.

The current host and worker explicitly support only EGO and SUPER. The worker
returns a reference at the current time and counts native trajectory messages;
it does not yet export a complete trajectory horizon for a downstream MPC.
The [first-release design](../docs/release-plan.md#通用原生方法接口首版要求)
requires a shared native-method interface with per-algorithm adapters. Container
launching remains one deployment option; future C++ integrations should reuse
the same host contract after their actual input/output capabilities are verified.


## 2026-09-29 五任务适配

当前默认名义速度20米/秒，navigation_v2时限300秒。悬停／跟踪／竞速使用显式时间参考到滚动目标适配器，完整数值和命令见`../docs/verification/final-acceptance/`。

`patches/ego-3d-goals.patch`保留交互目标的三维高度；预设导航目标路径保持原逻辑。`patches/super-control-initial-time.patch`仅在`DRONE_PLAYGROUND_CONTROL_TRANSFER=1`时改善亚米级轨迹的初始时间猜测，优化目标与约束保持原定义；导航设为0。setup.sh在固定来源上应用补丁再构建。每次新原生运行记录编译产物与相关来源文件SHA256。

地面竞速开始时统一位置控制器先以0.8米垂直目标接管，实际计时和任务判据照常推进；有效原生轨迹出现后交还规划器。备用控制步数、规划轨迹和指令数分别记录，零轨迹／零指令回合标记为执行失败。
