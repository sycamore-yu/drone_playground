# ADR-0012: sensor

**Status:** Accepted

**批准记录时间:** 2026-10-10T08:28:44Z。用户批准本轮全部建议并要求实施，包括快照采集。

**创建时间:** 2026-10-10T15:17:33+08:00

**最近核对时间:** 2026-10-10T10:30:23Z；实测版本为 MuJoCo 3.15.0、MuJoCo-LiDAR 0.3.5、JAX 0.11.2（RTX 4090 CUDA）。

**代码基线:** `v0.2@6ec50c7a441bbe169fef40acd34ae574b53428a9`

**工作分支:** `refactor/sensor-rendering`

**实现状态:** 功能分支已接入扫描资源、快照采集、相邻位姿插值、延迟交付与恢复身份。Depth/LiDAR × PPO/APG/SHAC 六组合完成 GPU 实际更新，完整更新吞吐实测提升，Depth 与 LiDAR 冻结策略完成八场景复测（见[验证记录](../validation.md)）。MuJoCo-LiDAR 原生求交替换仍未通过边界正确性门禁，保留原有唯一 `Scene.raycast()`，尚未合并到 `v0.2`。

## 背景

项目需要复用成熟传感器实现，并减少自有代码。当前 `Scene.raycast()` 与 MuJoCo-LiDAR JAX 内核都实现基本几何求交，长期并存会增加维护成本。当前 Depth/LiDAR 的采集梯度已被截断；Crazyflow 动力学、策略网络、状态观测和连续几何代价的梯度另行保留。

**技术研究链接：**[跨仓库求交与高保真 Review](../research/sensor-raycasting-fidelity-review.md)。
该研究区分第三方 `MjLidarJax` 和 MuJoCo 官方 `mjx.ray`；
复核 JAX/CPU 原生求交的约定差异、维护者公开 Issue，以及当前快照采集的真实性边界。
它是后续评估证据，不自动替换此前已批准的快照模式。

本次核查的 `assets/scenes/` 没有 Mesh 或高度场声明；`Scene` 当前仅接受基本几何体。回放机器人外观不构成本次训练需要 Mesh 求交的证据。

## 决策

### 一套求交实现；优先成熟实现但以验证结果为准

**当前决定（2026-10-10）：保留现有唯一的 JAX `Scene.raycast()`，LiDAR 与理想 Depth 均使用它求交。** MuJoCo-LiDAR 继续负责提供 MID360 非重复角度资源；`Scene.clearance()` 继续负责可微碰撞/净空损失。原因不是偏好自研，而是经实测，MuJoCo-LiDAR 0.3.5 的 `MjLidarJax` 和官方 MuJoCo 3.15.0 的 MJX-JAX `ray()` 均未通过当前场景的完整几何正确性与接入门禁。相机部分见 [ADR-0013](0013-render.md)。

只维护一条正式采集路径。不再把第三方求交替换当作本轮必须完成的发布目标；未来上游完整支持当前几何合同、批量 JIT、动态场景和 GPU 性能后，再**一次性替换并删除**自写 `_intersection`，不建设第二条永久后端、逐几何混用或自动回退。保留正式正确性测试和迁移结果，历史代码由 Git 保存。不得删除仍供碰撞和可微损失使用的 `Scene.clearance()` 及其距离计算。

不增加 Backend 注册系统，不同时安装 CPU、Taichi、Warp 多套实现，不复制上游源码。`pyproject.toml` 与 `pixi.lock` 固定 `mujoco-lidar==0.3.5`，当前只复用其扫描数据。原生计算核心仅由迁移检查脚本调用。

### 原上游替换提案（未启用）

原提案拟直接使用 `MjLidarJax` 而非 `MjLidarWrapper`；后者围绕宿主 `MjData` 管理可变状态，其批量接口调用 `np.asarray()`。原生核心确实接收几何位姿与射线数组，但其边界求交未满足当前协议。现在仅在迁移检查脚本中调用该核心，不在训练循环内使用它。

MID360 角度数据在初始化时加载一次。训练阶段以只读数组和每环境显式相位索引采样，不共享 `LivoxGenerator` 的 Python 可变游标。相位随局部 reset 和 checkpoint 保存、恢复。

使用 20,000 次查询/帧、10 Hz 的采集预算和 1,024 点的策略输入预算；有效回波数由场景决定。按完整 800,000 条角度序列连续取 20,000 条并循环，不对上游 24,000 条窗口另做重采样。首个 LiDAR 帧在 0.1 s 产生，使用表的首段；局部 reset 重置对应世界的相位。运行报告和 checkpoint 记录源文件 SHA-256、相位规则、快照语义及预处理预算。源表原样保留，包括约 −7.212°～52.164° 的角度范围；它提供扫描方向参考，不包含厂家逐点时间标定。

### 快照采集

在帧时间 `t_k`，使用机器人位姿和 `Scene.positions(t_k)` 计算整帧。每帧更新动态障碍物；各点的采集时间均为 `t_k`，可用时间为 `t_k + latency`。角度索引只表示扫描相位。

这项已批准的模型简化省去帧内运动畸变、长位姿历史、重复点云和去畸变。它与原逐射线模型的任务表现须单独比较；例如 0.1 秒内以 3 m/s 相对运动会移动 0.3 m，不能把忽略此运动得到的耗时变化计作纯后端加速。

`SensorObservation` 缩减为采样调度、最新测量、相位和交付状态。物理时钟继续由 Environment 持有；非整数频率的帧时刻可用相邻两次物理位姿插值，不保存逐射线长历史。测量延迟仍遵守已批准的 [ADR-0009](0009-delayed-data-in-episode-state.md)，队列属于环境实例。GRU 记忆仍属于策略，不移入 Sensor。

原 `pose_at` 参数、`pose_history`、`points_at_completion` 和 `pending_points` 已删除。只保留最近一次物理位姿、测量所属位姿及必要延迟队列；相邻四元数先统一符号再插值，避免 `q` 与 `−q` 产生零四元数。旧完整训练状态不通过新模板恢复，不增加迁移或兼容读取分支。冻结评测核对归档中的传感器身份；缺失或不一致时拒绝运行。固定策略跨模型实验须显式使用已有 `benchmark.allow_environment_change`，报告记录实际传感器身份。历史文件保持不变。

### 原生替换门禁的实测结果

`tools/check_sensor_migration.py` 对七个明确的边界案例逐项比较原生 JAX 核心与 `mujoco.mj_ray`。0.3.5 在其中六例不一致：四种实体内部起点返回 0 而非出射交点；平面背面产生额外命中；沿 Box 面平行发射的射线漏检。正常外部 Box 命中一致。这个比例只描述所选边界案例，不代表总体精度。

因此未执行求交替换和旧内核删除。`Scene`、碰撞及连续净空损失保持原实现，不通过改测试期望、增加自动回退或复制修补上游内核来绕过门禁。原生缺陷修复并通过原有几何测试、性能和任务验收后，才能重新考虑该替换。

### 官方 MJX-JAX `ray()` 替换评估（2026-10-10）

对已安装的官方 MuJoCo / MJX 3.15.0 直接检查 `mjx.ray` 和运行实测；详细记录见[验证文档](../validation.md)。结论是**当前不采用**：

- MJX-JAX `ray()` 的形状函数仅包含 Plane、Sphere、Capsule、Ellipsoid、Box、Mesh，**不含 Cylinder 和 HField**。现有 S01–S03、D01–D03、S06/D06 都含大量 Cylinder，直接替换会漏检核心导航障碍物；虽然 Mesh 可用，官方明确其 JAX 求交较慢，且目前场景没有 Mesh 需求。
- 当前 Navigation MJCF 同时含 Cylinder 和 Box，原样 `mjx.put_model()` 因不支持该碰撞组合而报 `NotImplementedError`，尽管调用方只想射线查询；这些场景也均没有材质（`nmat=0`），直接 `mjx.ray()` 因空 `mat_rgba` 索引在 JAX 编译时报错。
- 仅在`tmp/` 的独立实验中，复制 MuJoCo 模型并禁用候选的碰撞，给 MJX Model 提供占位材质数组，再手动以 `Scene.positions(t)` 更新 MJX Data 的 `geom_xpos`。这样动态移动 Box 的距离在 0/0.5/1 秒分别为 0.600/0.632/0.665 m，与原路径一致；不更新 Data 则产生陈旧错误测距。表明 JAX 状态对接可行，但它要求**第二份模型、显式同步与上游缺陷绕行**。
- 两世界各 256 射线的 JIT/`vmap` 诊断在 CPU 以已适配场景运行；GPU 默认浮点矩阵精度只得到 430/512 一致，明确设置 `JAX_DEFAULT_MATMUL_PRECISION=highest` 后才达到 512/512。用户的 GPU 同时有其它训练任务，所得 2 ms 量级小批量时间不构成独立高吞吐性能验收，也不能代表包含 Cylinder 的完整场景。

这些限制与纯 Crazyflow/JAX 的可微物理路径无关；传感器当前已截断梯度。**在上游解决 Cylinder、无材质求交及仅测距模型转换问题之前，维持现有 `Scene.raycast()` 的单实现结构更符合可靠性与维护成本目标。** 不为迁就 MJX-JAX 修改现有障碍物几何、增加模型适配层或引入混合后端。若未来确需复杂 Mesh，应单独评估官方 MJX-Warp/RenderContext，而非为当前基本几何场景提前增加依赖。

## 迁移与验收

求交替换、扫描方向变化、采样时间简化分别检查，避免把观测变化算作纯后端加速。测试中固定场景、射线、位姿、时刻和量程，对照官方 MuJoCo 查询。检查遮挡、未命中、量程边界、内部起点、可见几何和自体排除；边界语义差异必须解决或明确记录，不能静默漏掉物体。

原生接入须通过批量 JIT、局部 reset、延迟交付和完整恢复检查；APG/SHAC 中采集梯度保持截断，Actor/Crazyflow/净空代价梯度保持可用。PPO 使用同一测量合同。

快照方案单独验证静态/动态场景，使用固定策略和既定评测集检查任务表现。若需重新训练，使用 checkpoint validation 选模，冻结 benchmark 不用于反复调参。测试包括 3 m/s 与 20 m/s 的帧时刻、位姿和动态距离；这些数学检查不替代整回合飞行验收。正式任务门禁通过前，本功能分支不合并到集成分支。

性能比较使用相同工作负载，分开报告采集耗时、完整更新耗时及峰值显存；先预热，再同步 GPU 计时，交替执行并重复测量。只有正确性、既定任务验收和资源预算均无实质退化，才删除对应旧实现。不得通过缩小射线数、降低频率或简化时间模型掩盖后端退化。

## 取舍

原先希望通过复用第三方 JAX 求交器减少重复数学实现，避免跨运行时数据通道。但两项已测试的第三方/官方候选均有当前任务不可接受的缺陷，采用它们反而需要多余模型、副本和兼容补丁。现保留一套经过基准验证的项目 JAX 求交实现，继续复用官方 MuJoCo 的 MJCF 编译与 C `mj_ray` 测试参考，以及 MuJoCo-LiDAR 的 MID360 角度资源。项目仍负责 Crazyflow 的时间、状态和测量交付。

本次不扩展 Mesh 场景，不改变动力学、网络或损失，不引入 MJX-Warp。

## 依据

- [当前 Scene](../../src/drone_playground/simulation/scene.py)、[采集状态](../../src/drone_playground/simulation/observation.py)、[设备测量](../../src/drone_playground/simulation/sensors.py)。
- [MuJoCo-LiDAR JAX 核心](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/core_jax/mjlidar_jax.py)：本次读取的源文件 Git blob 为 `8ee01fa104b52d855162b48e9bfdae68b2226a6e`。
- [MuJoCo-LiDAR 包装器](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/lidar_wrapper.py)：本次读取的源文件 Git blob 为 `7089fceb2aa3769bdc214558873947732c7671df`。
- [扫描生成器](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/scan_gen.py)。
