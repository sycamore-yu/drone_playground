# ADR-0012: sensor

**Status:** Proposed

**批准时间:** 未批准具体实现；用户已明确要求替换式接入、验收后删除重复实现。

**创建及最近核对时间:** 2026-10-10T15:17:33+08:00

**代码基线:** `v0.2@6ec50c7a441bbe169fef40acd34ae574b53428a9`

**工作分支:** `refactor/sensor-rendering`

**实现状态:** 仅记录设计。尚未安装依赖、修改传感器、删除求交代码或运行性能与训练验收。

## 背景

项目需要复用成熟传感器实现，并减少自有代码。当前 `Scene.raycast()` 与 MuJoCo-LiDAR JAX 内核都实现基本几何求交，长期并存会增加维护成本。当前 Depth/LiDAR 的采集梯度已被截断；Crazyflow 动力学、策略网络、状态观测和连续几何代价的梯度另行保留。

本次核查的 `assets/scenes/` 没有 Mesh 或高度场声明；`Scene` 当前仅接受基本几何体。回放机器人外观不构成本次训练需要 Mesh 求交的证据。

## 提议决策

### 一套原生求交实现

使用 MuJoCo-LiDAR 的 `MjLidarJax` 原生计算核心。LiDAR 射线与针孔相机射线共用 `render_batch()`。保留 `Scene.raycast()` 这个已有窄入口，使现有调用方继续使用它，但将内部数学求交替换成上游调用。相机部分见 [ADR-0013](0013-render.md)。

只维护一条正式采集路径。旧、新实现只在功能分支的迁移检查中并存；通过验收后删除自写 `_intersection` 等求交实现及旧后端选择。保留正式正确性测试和迁移结果，历史代码由 Git 保存。不得删除仍供碰撞和可微损失使用的 `Scene.clearance()` 及其距离计算。

不增加通用 Backend 注册系统，不同时安装 CPU、Taichi、Warp 多套实现。不复制上游源码。实现阶段将实际验证的包版本固定到现有依赖锁文件；本 ADR 不将旧版 0.3.5 的角度资源使用经验当作完整原生后端的验收证据。

### 尽量原生，保留必要的 JAX 适配

直接使用 `MjLidarJax`，不把 `MjLidarWrapper` 放进训练循环。后者围绕宿主 `MjData` 管理可变状态，其批量返回接口调用 `np.asarray()`。原生 JAX 核心接收几何位姿与射线数组，适合现有批量环境。

MID360 角度数据在初始化时加载一次。训练阶段以只读数组和每环境显式相位索引采样，不共享 `LivoxGenerator` 的 Python 可变游标。相位随局部 reset 和 checkpoint 保存、恢复。

继续使用已有 20,000 点/帧与 10 Hz 的设备预算、1,024 点的策略输入预算。24,000 是上游生成器的默认窗口长度，不要求把每个窗口强制重采样为 20,000。建议按完整角度序列顺序每帧取 20,000 条并循环，记录源文件哈希和相位规则。该序列提供扫描方向参考；其原始逐点时间标定仍需来源证据，不能据此宣称厂家标定的完整设备仿真。

### 采集时间：独立的快照模型提案

建议将本分支目标收敛为瞬时帧采集：在帧时间 `t_k`，使用机器人位姿和 `Scene.positions(t_k)` 计算整帧。每一帧都更新动态障碍物，因此仍支持动态导航。各点的采集时间均为 `t_k`，可用时间为 `t_k + latency`；角度索引不再冒充逐点物理时间。

这会改变当前逐射线采样语义，属于待确认的模型简化，不能随求交替换自动生效。它省去扫描期间的运动畸变模拟、长位姿历史和去畸变处理。平移运动量尺度为相对速度乘扫描时长；例如 0.1 秒内以 3 m/s 相对运动会移动 0.3 m。这是量级示例，不是测距误差上界。

`SensorObservation` 缩减为采样调度、最新测量、相位和交付状态。物理时钟继续由 Environment 持有；非整数频率的帧时刻可用相邻两次物理位姿插值，不保存逐射线长历史。测量延迟仍遵守已批准的 [ADR-0009](0009-delayed-data-in-episode-state.md)，队列属于环境实例。GRU 记忆仍属于策略，不移入 Sensor。

在快照方案获批前，实施阶段保持当前逐点时间合同。需要保留该合同时，可在窄适配函数中按实际射线时间调用上游核心，不为此 fork 求交内核或新增永久模式集合。

## 迁移与验收

求交替换、扫描方向变化、采样时间简化分别检查，避免把观测变化算作纯后端加速。测试中固定场景、射线、位姿、时刻和量程，对照官方 MuJoCo 查询。检查遮挡、未命中、量程边界、内部起点、可见几何和自体排除；边界语义差异必须解决或明确记录，不能静默漏掉物体。

原生接入须通过批量 JIT、局部 reset、延迟交付和完整恢复检查；APG/SHAC 中采集梯度保持截断，Actor/Crazyflow/净空代价梯度保持可用。PPO 使用同一测量合同。

快照方案单独验证静态/动态场景，使用固定策略和既定评测集检查任务表现。若改变了训练分布，重新训练使用 checkpoint validation 选模，冻结 benchmark 不用于反复调参。验收必须覆盖声明的速度范围；失败则停止该简化，不把旧、新模型作为两个永久产品后端发布。

性能比较使用相同工作负载，分开报告采集耗时、完整更新耗时及峰值显存；先预热，再同步 GPU 计时，交替执行并重复测量。只有正确性、既定任务验收和资源预算均无实质退化，才删除对应旧实现。不得通过缩小射线数、降低频率或简化时间模型掩盖后端退化。

## 取舍

复用原生 JAX 核心能减少重复数学实现，并避免新增跨运行时数据通道。项目仍负责 Crazyflow 的时间、状态和测量交付，因为 MuJoCo-LiDAR 的几何查询接口没有提供这些环境职责。

本提案不扩展 Mesh 场景、不改变动力学或损失，也不同时引入 MJX-Warp。新需求和实测瓶颈出现后再决定下一次替换。

## 依据

- [当前 Scene](../../src/drone_playground/simulation/scene.py)、[采集状态](../../src/drone_playground/simulation/observation.py)、[设备测量](../../src/drone_playground/simulation/sensors.py)。
- [MuJoCo-LiDAR JAX 核心](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/core_jax/mjlidar_jax.py)：本次读取的源文件 Git blob 为 `8ee01fa104b52d855162b48e9bfdae68b2226a6e`。
- [MuJoCo-LiDAR 包装器](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/lidar_wrapper.py)：本次读取的源文件 Git blob 为 `7089fceb2aa3769bdc214558873947732c7671df`。
- [扫描生成器](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/scan_gen.py)。
