# 执行账本：.scratch/pointcloud-paper/implementation.md

2026-09-28 创建，主会话直接执行。
来源冻结：论文8页；[24] DiffPhysDrone@271936190b5c2a5e760230e718fa6b167718a5bf。
工作树pointcloud-paper，基线810f7d3，Navigation8已冻结。RTX4090空闲，磁盘3.5T可用。
独立性：现有refactor/native-planner-runtime工作树有进行中改动，本任务使用独立分支。
裁决：作者源码搜索无匹配，公开重建假设，不声明未知参数与作者一致。
裁决：保留50000更新完整目标；训练进行中的里程碑只报告实际预算与状态。
接口预检：物理/传感组件→PointCloudTask→训练与评测；PointCloudPolicy共享训练/冻结推理；
共享SceneBank几何用于训练损失、传感器、Navigation8碰撞与回放，避免场景不一致。
下一步：任务一测试先行。

基线：SHAC、深度、LiDAR共31通过、1失败。失败test_batch_environments_are_isolated_and_reset_clears_the_phase
仍假设ID0/3是不同随机几何；Navigation8的固定ID展开使两者都为S01。后续改用不同场景分组索引验证真实隔离。
任务一RED：新物理/传感模块六项明确因缺失模块/球体支持失败，见tmp/pointcloud/physics-red.log。
共享几何复核新增两条边界：轴对齐盒的平行射线有效命中、静态障碍时间导数安全。

任务一/二/训练核心：33项组件回归通过，含原深度与修正固定场景索引的LiDAR隔离测试。
新增8项物理/传感、5项网络/目标/场景、4项训练/恢复测试均经历RED→GREEN。
CPU保存恢复后的优化器、随机数、下一参数更新逐元素一致；编码器参数梯度有限非零。
GPU完整形状工程探测paper-pointcloud-throughput-seed0-v1真实3更新/15360交互，退出0，状态paused。
首次更新含编译28.858秒，稳定1.830秒/更新，独立开发损失22562.32→9.0534；仅作为工程结果。
按该吞吐50000更新约25.4小时。正式训练分可恢复阶段执行，目标预算仍为256000000交互。
下一动作：提交当前核心模块；启动正式首阶段1000更新，同时完成独立Navigation8评测与回放模块。
