# ADR-0012: sensor

**Status:** Accepted

**批准记录时间:** 2026-10-10T08:28:44Z。用户批准本轮全部建议并要求实施，包括快照采集。

**创建时间:** 2026-10-10T15:17:33+08:00

**最近核对时间:** 2026-10-10T08:52:38Z；实测版本为 MuJoCo 3.15.0、MuJoCo-LiDAR 0.3.5、JAX 0.11.2。

**代码基线:** `v0.2@6ec50c7a441bbe169fef40acd34ae574b53428a9`

**工作分支:** `refactor/sensor-rendering`

**实现状态:** 功能分支已接入扫描资源、快照采集、相邻位姿插值、延迟交付与恢复身份。原生求交替换未通过正确性门禁，保留原有唯一求交路径；GPU 性能及冻结任务验收未完成。具体证据见[验证记录](../validation.md)。

## 背景

项目需要复用成熟传感器实现，并减少自有代码。当前 `Scene.raycast()` 与 MuJoCo-LiDAR JAX 内核都实现基本几何求交，长期并存会增加维护成本。当前 Depth/LiDAR 的采集梯度已被截断；Crazyflow 动力学、策略网络、状态观测和连续几何代价的梯度另行保留。

本次核查的 `assets/scenes/` 没有 Mesh 或高度场声明；`Scene` 当前仅接受基本几何体。回放机器人外观不构成本次训练需要 Mesh 求交的证据。

## 决策

### 一套原生求交实现

目标是使用 MuJoCo-LiDAR 的 `MjLidarJax` 原生计算核心。LiDAR 射线与针孔相机射线共用 `render_batch()`。保留 `Scene.raycast()` 这个已有窄入口；通过下述门禁后替换内部数学求交。相机部分见 [ADR-0013](0013-render.md)。

只维护一条正式采集路径。旧、新实现只在功能分支的迁移检查中并存；通过验收后删除自写 `_intersection` 等求交实现及旧后端选择。保留正式正确性测试和迁移结果，历史代码由 Git 保存。不得删除仍供碰撞和可微损失使用的 `Scene.clearance()` 及其距离计算。

不增加 Backend 注册系统，不同时安装 CPU、Taichi、Warp 多套实现，不复制上游源码。`pyproject.toml` 与 `pixi.lock` 固定 `mujoco-lidar==0.3.5`，当前只复用其扫描数据。原生计算核心仅由迁移检查脚本调用。

### 尽量原生，保留必要的 JAX 适配

直接使用 `MjLidarJax`，不把 `MjLidarWrapper` 放进训练循环。后者围绕宿主 `MjData` 管理可变状态，其批量返回接口调用 `np.asarray()`。原生 JAX 核心接收几何位姿与射线数组，适合现有批量环境。

MID360 角度数据在初始化时加载一次。训练阶段以只读数组和每环境显式相位索引采样，不共享 `LivoxGenerator` 的 Python 可变游标。相位随局部 reset 和 checkpoint 保存、恢复。

使用 20,000 次查询/帧、10 Hz 的采集预算和 1,024 点的策略输入预算；有效回波数由场景决定。按完整 800,000 条角度序列连续取 20,000 条并循环，不对上游 24,000 条窗口另做重采样。首个 LiDAR 帧在 0.1 s 产生，使用表的首段；局部 reset 重置对应世界的相位。运行报告和 checkpoint 记录源文件 SHA-256、相位规则、快照语义及预处理预算。源表原样保留，包括约 −7.212°～52.164° 的角度范围；它提供扫描方向参考，不包含厂家逐点时间标定。

### 快照采集

在帧时间 `t_k`，使用机器人位姿和 `Scene.positions(t_k)` 计算整帧。每帧更新动态障碍物；各点的采集时间均为 `t_k`，可用时间为 `t_k + latency`。角度索引只表示扫描相位。

这项已批准的模型简化省去帧内运动畸变、长位姿历史、重复点云和去畸变。它与原逐射线模型的任务表现须单独比较；例如 0.1 秒内以 3 m/s 相对运动会移动 0.3 m，不能把忽略此运动得到的耗时变化计作纯后端加速。

`SensorObservation` 缩减为采样调度、最新测量、相位和交付状态。物理时钟继续由 Environment 持有；非整数频率的帧时刻可用相邻两次物理位姿插值，不保存逐射线长历史。测量延迟仍遵守已批准的 [ADR-0009](0009-delayed-data-in-episode-state.md)，队列属于环境实例。GRU 记忆仍属于策略，不移入 Sensor。

原 `pose_at` 参数、`pose_history`、`points_at_completion` 和 `pending_points` 已删除。只保留最近一次物理位姿、测量所属位姿及必要延迟队列；相邻四元数先统一符号再插值，避免 `q` 与 `−q` 产生零四元数。旧完整训练状态不通过新模板恢复，不增加迁移或兼容读取分支。冻结评测核对归档中的传感器身份；缺失或不一致时拒绝运行。固定策略跨模型实验须显式使用已有 `benchmark.allow_environment_change`，报告记录实际传感器身份。历史文件保持不变。

### 原生替换门禁的实测结果

`tools/check_sensor_migration.py` 对七个明确的边界案例逐项比较原生 JAX 核心与 `mujoco.mj_ray`。0.3.5 在其中六例不一致：四种实体内部起点返回 0 而非出射交点；平面背面产生额外命中；沿 Box 面平行发射的射线漏检。正常外部 Box 命中一致。这个比例只描述所选边界案例，不代表总体精度。

因此未执行求交替换和旧内核删除。`Scene`、碰撞及连续净空损失保持原实现，不通过改测试期望、增加自动回退或复制修补上游内核来绕过门禁。原生缺陷修复并通过原有几何测试、性能和任务验收后，才能完成该替换。

## 迁移与验收

求交替换、扫描方向变化、采样时间简化分别检查，避免把观测变化算作纯后端加速。测试中固定场景、射线、位姿、时刻和量程，对照官方 MuJoCo 查询。检查遮挡、未命中、量程边界、内部起点、可见几何和自体排除；边界语义差异必须解决或明确记录，不能静默漏掉物体。

原生接入须通过批量 JIT、局部 reset、延迟交付和完整恢复检查；APG/SHAC 中采集梯度保持截断，Actor/Crazyflow/净空代价梯度保持可用。PPO 使用同一测量合同。

快照方案单独验证静态/动态场景，使用固定策略和既定评测集检查任务表现。若需重新训练，使用 checkpoint validation 选模，冻结 benchmark 不用于反复调参。测试包括 3 m/s 与 20 m/s 的帧时刻、位姿和动态距离；这些数学检查不替代整回合飞行验收。正式任务门禁通过前，本功能分支不合并到集成分支。

性能比较使用相同工作负载，分开报告采集耗时、完整更新耗时及峰值显存；先预热，再同步 GPU 计时，交替执行并重复测量。只有正确性、既定任务验收和资源预算均无实质退化，才删除对应旧实现。不得通过缩小射线数、降低频率或简化时间模型掩盖后端退化。

## 取舍

复用原生 JAX 核心能减少重复数学实现，并避免新增跨运行时数据通道。项目仍负责 Crazyflow 的时间、状态和测量交付，因为 MuJoCo-LiDAR 的几何查询接口没有提供这些环境职责。

本次不扩展 Mesh 场景，不改变动力学、网络或损失，不引入 MJX-Warp。

## 依据

- [当前 Scene](../../src/drone_playground/simulation/scene.py)、[采集状态](../../src/drone_playground/simulation/observation.py)、[设备测量](../../src/drone_playground/simulation/sensors.py)。
- [MuJoCo-LiDAR JAX 核心](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/core_jax/mjlidar_jax.py)：本次读取的源文件 Git blob 为 `8ee01fa104b52d855162b48e9bfdae68b2226a6e`。
- [MuJoCo-LiDAR 包装器](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/lidar_wrapper.py)：本次读取的源文件 Git blob 为 `7089fceb2aa3769bdc214558873947732c7671df`。
- [扫描生成器](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/scan_gen.py)。
