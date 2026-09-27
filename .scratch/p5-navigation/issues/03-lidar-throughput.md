# P5-03：MID360与吞吐冻结

Type: task
Status: resolved
Blocked by: P5-01；总吞吐表另依赖P5-02
Engineering: passed
Experiment: throughput-measured
Quality: separate (P5-07)
Owner: main-session
Session: p5-main
Run: p5_static_lidar_ppo throughput probe

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

复用MuJoCo-LiDAR建立点云观测与回放；测两种传感器在1/16/128环境的开销和训练内存。

## 验收

MID360扫描方向/相位/点时间明确；动态几何更新和批量隔离通过；按67108864交互推算墙钟预算并提交具体冻结表。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

实现：`tasks/sensors/lidar.py`。扫描方向**复用** `mujoco-lidar==0.3.5` 的 MID360 图案
（800000 条预计算射线、每次扫描 24000 条、33 个可用窗口、方位 360°、俯仰 -7.21°..52.16°），
窗口在宿主机一次物化为静态表，扫描相位因此成为回合状态而不是第三方生成器里的可变游标，
重置即回到地平线起点。训练策略每帧取 120 点（步长 200，索引规则写入标定记录）。
逐点相对时间按角度进度记录；首版几何按**瞬时扫描**求值，该选择显式写入标定。
世界系点云由 `point_cloud_message` 实际做 `R_world_from_body` 变换后给出，供原生规划器使用。

验收检查 `tests/test_lidar_sensor.py` 共 12 项通过：图案覆盖与窗口非重复、单位方向与
Livox 约定、逐点时间与周期、三组位姿下与 `mujoco.mj_ray` 及 **MuJoCo-LiDAR 自带
`MjLidarJax` 批量射线后端**的一致、策略窗口是源窗口的真实子集、世界系变换关系、
批量环境隔离、10 Hz 相位与重置清空、量程与无效点约定、自机不可见。

吞吐表由 `scripts/p5_throughput_probe.py` 实测写入 `docs/verification/p5-throughput.json`，
覆盖 1/16/128 环境、状态/深度/点云三种配置，含编排耗时、单步耗时、传感器单帧耗时
与按 8388608 交互推算的单单元墙钟。

**已实测的吞吐事实**：真实 PPO 训练在 128 环境、1048576 交互下
`training/sps=21795.7`（深度单元）；环境单步在 128 环境下约 0.75 毫秒、
约 170000 交互/秒，说明瓶颈在策略更新而不在传感器。
