# ADR-0010：训练专用特权 PPO Critic

**Status:** Accepted · 2026-10-09

**Implementation:** 窄入口已加入，真实更新、完整恢复、Actor 推理隔离与配置校验共五项定向回归通过；完整 CPU 回归 239 项通过。Depth 种子 0 已启动，尚无特权 PPO 收敛结果。

用户在持续推进从零训练时明确授权启用 AsymmetricPPO，并参考 DiffAero 的训练设计。普通 PPO 的状态 Critic 与传感器 Critic 已有长时间未达标的证据，允许独立测试更完整的仿真信息；这不表示特权方案必然改善成功率。

复用当前 PPO 的更新、循环 Actor、GAE 和 checkpoint 流程，通过 `learning.options.critic_uses_privileged=true` 选择独立价值网络。该选项仅用于 Depth/LiDAR Navigation PPO，与 `critic_uses_sensor` 互斥。其默认值为 false，旧训练状态可继续恢复。

训练 Critic 获得原机体特征及真实全局位置、目标位移、四元数、线/角速度、前一动作、归一化时间、带符号净空和 26 条世界坐标系理想几何射线。射线覆盖 `{-1,0,1}³` 的非零方向，最大距离 40 m，没有传感器的噪声、延迟和视场限制。所有场景使用同一固定维度和尺度；不是把不同数量的障碍物直接铺平。

Actor 始终只读取现有机体和真实传感器观测。冻结文件只导出 Actor；评测仍使用原设备采样和预处理，不调用特权入口。不得把此结果称为普通同观测 PPO 消融，也不得把训练特权信息当作部署输入。

参考 [DiffAero AsymmetricPPO](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/algo/PPO.py) 与 [ObstacleAvoidance.get_state](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/env/obstacle_avoidance.py)。本实现借鉴输入分离原则，仍使用 Crazyflow 与当前任务，并非完整移植 DiffAero 的网络、物理或奖励。
