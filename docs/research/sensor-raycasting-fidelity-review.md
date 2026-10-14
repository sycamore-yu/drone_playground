# Sensor ray casting：成熟实现、语义一致性与高保真评估

> **研究时间：**2026-10-10
> **性质：**基于第一方源码与本仓库实测的技术研究；不是新增后端的实现规格。
> **对应决策：**[ADR-0012 sensor](../adr/0012-sensor.md)、
> [ADR-0013 render](../adr/0013-render.md)；运行证据见[validation](../validation.md)。
> **实测环境：**MuJoCo 3.15.0、MuJoCo-LiDAR 0.3.5、JAX 0.11.2、RTX 4090；
> 分支 `refactor/sensor-rendering`。下面的架构事实与工程判断分别标注。

## 结论

**当前仅保留一个正式求交实现——JAX `Scene.raycast()`，由 MuJoCo 的 MJCF
编译结果初始化，并持续用官方 `mj_ray()` / `mj_multiRay()` 作独立正确性参考。**
MID360 扫描方向继续复用 MuJoCo-LiDAR 的角度表，理想 Depth 共用同一求交接口。
此项不是偏好“自研”，而是现有场景含有大量 Cylinder，实测两个 JAX 上游候选
不满足完整语义。引入它们需要额外模型同步、混合求交甚至复制补丁，会增加维护成本。

**若研究目标从理想几何避障提升为高速 Sim-to-Real 感知，现有“每帧所有射线共用一个时刻”
的快照模型比求交库是否来自第三方更值得重新审查。** 这与选择哪一种射线几何内核
是两个独立问题；本研究不自动推翻已批准的快照合同，也不修改训练代码。

## 1. 三个不同层次：扫描、求交、成像

1. **Scan pattern（扫描模式）：**给出每条射线方向、采样顺序、时间偏移、
   设备内外参与发射视场。扫描模式正确不保证命中正确。
2. **Ray intersection（射线求交）：**给定场景几何、位姿、射线，
   返回第一表面命中距离、遮挡关系和有效性。CPU、JAX、Warp 的同名方法
   不一定有相同的内部起点、平面背面和边界切线约定。
3. **Sensor measurement（传感器测量）：**在求交基础上定义量程、噪声、
   失效回波、材质/反射率、运动畸变、时间戳以及相机光轴深度。
   原生几何引擎通常不会自动提供完整硬件测量模型。

现有项目分别在
[`sensors.py`](../../src/drone_playground/simulation/sensors.py)、
[`scene.py`](../../src/drone_playground/simulation/scene.py)、
[`observation.py`](../../src/drone_playground/simulation/observation.py)
承载这些职责；训练算法并不依赖传感器反向传播。

## 2. 代表性 Robot Learning 仓库如何求交（第一方源码）

| 工程 | 传感器/求交的实际路径 | 用作本项目成熟替换件？ |
|---|---|---|
| **P2M (HKU)** | 训练端：Isaac Lab `RayCaster` + Bpearl 网格射线，移动圆柱与墙体另以 Torch 解析近似计算命中并合并；ROS 推理端：`raycast.cpp` 把已有地图点云按角度分桶，选各桶最近点，**不是对 Mesh 逐射线求交**。 | **否**；很好的“静态求交+动态对象近似”案例，但不是 JAX 完整几何替换。 |
| **NavRL** | Isaac/Orbit `RayCaster` + `BpearlPatternCfg`；训练配置的 `mesh_prim_paths` 指向地面，动态物体信息在策略观测中另以位置/速度/尺寸显式表示。 | **否**；对任务观察设计有参考，不是 MID360 设备扫描或完整动态 Mesh LiDAR。 |
| **DiffAero** | 自写 `torch.jit.script` 解析 Sphere、Cube、Ground Plane 射线求交；`ObstacleAvoidanceRenderer`（Taichi）主要服务显示/录像，实际感知通过 `utils/sensor.py`。 | **否**；和我们目前的“解析图元求交、奖励独立于传感器”设计高度相似。 |
| **MuJoCo Playground** | 使用官方 MJX 仿真；README 明确以官方 **MJWarp Batch Renderer** 支持视觉训练。没有提供可直接替换 MID360 采集的项目统一求交 API。 | **参考官方成像集成方式**；不因名称相似就把 MJX-JAX `ray` 视为通用 LiDAR。 |
| **Omni-Perception** | Warp Kernel 调用 `wp.mesh_query_ray()` 对已注册三角网格作 BVH 求交；设备扫描模式由独立生成器/表提供，Torch↔Warp 转换。 | **有价值的未来 Mesh/MID360 性能参考**，但复用需要场景 Mesh/BVH 与 JAX↔Warp 数据合同，不是当前最小实现。 |
| **Unitree 官方 `unitree_rl_mjlab` + `mjlab`** | 配置 `RayCastSensorCfg`；底层 `mjlab.sensor.RaycastSensor` 使用 **MuJoCo Warp `rays`、BVH**，另有独立的 `CameraSensor`/Depth 路径；物理仿真本身由 MuJoCo Warp 提供。 | **成熟的传感器架构参考**，但其 Torch/Warp 场景生命周期不能直接嵌入现有纯 JAX `lax.scan`。 |
| **社区 MID360-G1 fork** | 复制 `terrain_scan` 传感器，使用 `OffsetGridPatternCfg` 与 `envs_mdp.height_scan`，施加视场 Mask、随机漏点、时间记忆。 | **否**；是 *MID360-like sparse height map*，不是真正 3D 非重复 Livox 光路。 |
| **MuJoCo-LiDAR** | CPU 后端直接使用 MuJoCo 官方 `mj_multiRay`；JAX 后端另写解析相交函数；Warp/Taichi 后端为各自实现。 | **扫描表可复用**；CPU 原生求交是可靠参考，但不能直接并入 GPU `jax.jit`；JAX 几何语义未达本项目门禁。 |
| **官方 MJX-JAX `ray()`** | MuJoCo 官方实现，但 3.15.0 的 JAX 分派不含 **Cylinder/HField**，`put_model` 还会校验碰撞函数。 | **当前不能取代**含大量 Cylinder 的 Navigation 场景。 |
| **官方 MuJoCo Warp / NVIDIA Warp** | 官方 BVH、动态 refit、Mesh 与深度渲染；Warp 单独提供 `mesh_query_ray`。 | **后续复杂 Mesh 场景的最优先研究候选**，必须先验证 GPU 大批量与 Crazyflow/JAX 交互成本。 |
| **Open3D `RaycastingScene`** | 成熟的三角网格射线第一交点与距离查询，官方支持 CPU 和实验性 SYCL GPU；有批量 `cast_rays`、`compute_signed_distance`，但不是在我们的 CUDA/JAX 训练图中原生执行。 | **适合作为离线 Mesh 真值/验证工具**，不是当前 500 Hz + JAX GPU 实时采集的优先实现。 |
| **Embree（RenderKit）** | 成熟的 C/C++ 射线追踪内核，主要面向 CPU，亦有 Intel SYCL 相关路径；需要自行管理几何与绑定。 | **可作为大型 Mesh 离线 CPU 参考**，无法不加适配直接替换 JAX GPU 求交。 |

原始源码（按 2026-10-10 检查时的提交固定）：

- **P2M `6aa1f7c`：**[Isaac 训练环境](https://github.com/arclab-hku/P2M/blob/6aa1f7cf464a158b1464fad314a77ad23ff2bcf5/resources/envs/single/env.py#L197-L223)、
  [动态圆柱/墙体近似](https://github.com/arclab-hku/P2M/blob/6aa1f7cf464a158b1464fad314a77ad23ff2bcf5/resources/envs/single/env_utils.py#L127-L224)、
  [ROS 点云角度桶](https://github.com/arclab-hku/P2M/blob/6aa1f7cf464a158b1464fad314a77ad23ff2bcf5/src/lidar/src/raycast.cpp#L96-L188)。
- **NavRL `3725bcc`：**[Isaac `RayCaster` 和动态障碍物观测](https://github.com/Zhefan-Xu/NavRL/blob/3725bcc2e7c1be4ecf1455d922299ae85042603a/isaac-training/training/scripts/env.py#L46-L62)、
  [动态对象在观察中单独表示](https://github.com/Zhefan-Xu/NavRL/blob/3725bcc2e7c1be4ecf1455d922299ae85042603a/isaac-training/training/scripts/env.py#L479-L530)。
- **DiffAero `291ea141`：**[Torch 解析求交](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/utils/sensor.py#L23-L183)、
  [独立的显示渲染器](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/utils/render.py)、
  [感知采集调用与观测梯度](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/env/obstacle_avoidance.py#L76-L142)。
- **MuJoCo Playground `d59156e`：**[README：官方 Warp 视觉路径](https://github.com/google-deepmind/mujoco_playground/blob/d59156e099a8785dd58fbce2e25a6ee1f7b5800c/README.md)。
- **Omni-Perception `a1059ae`：**[Warp mesh Kernel](https://github.com/aCodeDog/OmniPerception/blob/a1059ae3ffb91ebea2854f8633a28027a0477d1c/LidarSensor/LidarSensor/sensor_kernels/lidar_kernels_warp.py#L7-L50)、
  [MID360 模式与传感器](https://github.com/aCodeDog/OmniPerception/blob/a1059ae3ffb91ebea2854f8633a28027a0477d1c/LidarSensor/LidarSensor/lidar_sensor.py#L71-L88)；
  [作者公布的性能及代码完成度](https://github.com/aCodeDog/OmniPerception/blob/a1059ae3ffb91ebea2854f8633a28027a0477d1c/README.md)。
- **mjlab `033ae22`：**[官方 Warp 射线传感器实现](https://github.com/mujocolab/mjlab/blob/033ae22a2c7a30a25a6fa77b16c113ed88dd1b55/src/mjlab/sensor/raycast_sensor.py)、
  [相机/Depth 实现](https://github.com/mujocolab/mjlab/blob/033ae22a2c7a30a25a6fa77b16c113ed88dd1b55/src/mjlab/sensor/camera_sensor.py)、
  [射线模式与多帧传感器文档](https://github.com/mujocolab/mjlab/blob/033ae22a2c7a30a25a6fa77b16c113ed88dd1b55/docs/source/sensors/raycast_sensor.rst)。
  [Unitree 原仓库配置](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/config/g1/env_cfgs.py)。
- **社区 Unitree fork `01cbe73`：**[`terrain_scan_mid360` 配置](https://github.com/1YI-DING/unitree_rl_mjlab_MID360-LiDAR/blob/01cbe731b86a33e1c36afeed5fdd6f44d0afccf6/src/tasks/velocity/config/g1/env_cfgs.py#L464-L528)、
  [MID360-like height scan](https://github.com/1YI-DING/unitree_rl_mjlab_MID360-LiDAR/blob/01cbe731b86a33e1c36afeed5fdd6f44d0afccf6/src/tasks/velocity/mdp/observations.py#L58-L122)。
- **官方 GPU 引擎：**[MuJoCo Warp 批量 BVH/Depth](https://mujoco.readthedocs.io/en/latest/mjwarp/#batch-rendering)、
  [MJX `refit_bvh`/渲染集成限制](https://mujoco.readthedocs.io/en/latest/mjx.html#batch-rendering)、
  [NVIDIA `mesh_query_ray` API](https://nvidia.github.io/warp/v1.17/language_reference/_generated/warp.mesh_query_ray.html)；
  [Warp/JAX 零拷贝与 FFI 文档](https://nvidia.github.io/warp/v1.17/user_guide/interoperability.html)。
- **其它通用 Mesh 实现：**
  [Open3D RaycastingScene 官方 API](https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html)
  （CPU/SYCL，Mesh raycasts 和距离查询）；
  [Embree 官方仓库](https://github.com/RenderKit/embree)
  （CPU/SYCL 射线追踪 Kernel）。二者都不提供与当前 Crazyflow/JAX 一体化的
  CUDA 训练后端，未纳入本轮本机数值/性能验收。

## 3. 为什么第三方 `MjLidarJax` 与 MuJoCo 不一致？

**先纠正名称：`MjLidarJax` 不是 MuJoCo 官方的类。**
它由 [discoverse-dev/MuJoCo-LiDAR](https://github.com/discoverse-dev/MuJoCo-LiDAR)
独立实现；官方 MuJoCo 的 CPU `mj_ray/mj_multiRay`、官方 MJX-JAX `ray()`
与该第三方函数不是同一求交内核。第三方包本身的
[`MjLidarCPU`](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/e6b879df4b96daaed30ef10fe02650d29048fa9b/src/mujoco_lidar/core_cpu/mjlidar_cpu.py)
也直接转调用 `mj_multiRay`，而其
[`core_jax/geometry.py`](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/e6b879df4b96daaed30ef10fe02650d29048fa9b/src/mujoco_lidar/core_jax/geometry.py)
是单独实现的解析函数。这就是同包多后端可能存在不同语义的根本原因。

与官方 `mujoco.mj_ray`、项目现有 `Scene.raycast` 的七项边界实测如下。
未命中时 MuJoCo/第三方返回 `-1`，项目按合同用 `max_range=40 m` 表示无回波；
因此比较时须统一未命中语义，不把 `-1` 对 `40` 算作错误：

| 案例 | 官方 MuJoCo | 项目 `Scene.raycast` | 第三方 `MjLidarJax` |
|---|---:|---:|---:|
| 球心向外，半径 1 m | 1 m | 1 m | 0 m |
| Box 内向外 | 1 m | 1 m | 0 m |
| Cylinder 内沿轴向 | 2 m | 2 m | 0 m |
| Capsule 内沿轴向 | 3 m | 3 m | 0 m |
| Plane 背面向上 | 未命中 | 未命中（40 m 截断） | 1 m |
| 射线沿 Box 边界平行 | 1 m | 1 m | 未命中 |
| Box 外部正常命中 | 1 m | 1 m | 1 m |

**不是“第三方全部算法都有错误”，而是三个具体差异：**

- **内部起点的语义不一致。** JAX 源码在 Sphere 内部明确写了
  `is_inside` 时返回 `0.0`，Box 在内部时同样采用 0；
  项目合同和官方 C 求交返回沿正向射线的离开表面距离。
  第三方注释称“MuJoCo usually returns 0 if inside”，与以上安装版本的实测不一致。
- **Plane 面向不一致。** 第三方 `ray_plane_intersection` 只查正距离与边界，
  没有 MuJoCo 对单面 Plane 的正面射线判断。
- **Box 平行分量数值处理。** 第三方用
  `1/(direction + 1e-10*sign(direction))`，零方向分量仍为零分母；
  导致选定边界平行射线漏检。项目实现显式分离平行轴。

这些都是可复现的**合同差异/边界数值问题**，七例是专门选取的边界测试，
**不能解读为第三方实际随机射线 6/7 都算错**。

### 维护者是否知道？

截至 2026-10-10，对该仓库公开的 Issue 与 PR 列表进行了逐项检查：
**未发现维护者明确承认或修复上述 JAX 内部起点、背面 Plane 和 Box 平行问题的讨论。**
因而不能认定维护者知道，也不能认定他们不知道——只能说公开证据不足。

项目已处理过另一后端的问题：
[`#23` Warp 无限平面 BVH AABB 过薄导致漏检](https://github.com/discoverse-dev/MuJoCo-LiDAR/pull/23)，
这是 2026-09-16 的 Warp 修复，**并非** JAX 的 Plane 反面问题。
此外，官方第三方仓库
[`tests/test_integration.py`](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/e6b879df4b96daaed30ef10fe02650d29048fa9b/tests/test_integration.py)
虽然测试名为 `test_backend_consistency`，实际上只断言
`len(ranges) == len(theta)`，未比较不同后端返回的逐点距离；
这可以解释为何该类差异没有被现有集成测试覆盖。
**尚未代用户提交 Issue**；若未来需要上游修复，可附
[`tools/check_sensor_migration.py`](../../tools/check_sensor_migration.py)
的复现输入和参考距离，并明确期望遵循哪一类射线合同。

## 4. 官方 MJX-JAX `ray()` 为什么也不适合直接替换？

这里需要与第三方 `MjLidarJax` 明确区分。
已安装的官方 MuJoCo/MJX 3.15.0：

- `mujoco.mjx._src.ray._RAY_FUNC` 仅有 Plane、Sphere、Capsule、
  Ellipsoid、Box、Mesh，**没有 Cylinder / Heightfield**；
- S01、S02、S03 分别有 **41/45、81/85、162/166** 个圆柱图元；
  动态场景 D01、D02、D03、D06 也含大量圆柱；
- 直接 `mjx.put_model` 会因为场景中不支持的 Cylinder–Box 碰撞组合失败，
  即使只计划调用求交；无材质的 `nmat=0` 模型还会触发 `mat_rgba` 空索引问题；
- 临时测试中，复制模型、只对候选禁用碰撞、加虚拟材质后，能够通过
  `d.replace(geom_xpos=Scene.positions(t))` 让**移动 Box** 测距与现有路径一致。
  但这必须维护第二份场景、手动位姿同步，而且始终不能求 Cylinder。

在实验性 2 world × 256 rays 上，CPU 小批量看似较快，
GPU 则约同速；GPU 默认矩阵乘法精度仅 430/512 一致，
设为 `highest` 后为 512/512。此例没有覆盖圆柱命中，
测试时 GPU 有别的训练进程，因此**不能宣称官方 MJX-JAX 性能更优**。

复现产物：`tmp/mjx-ray-eval/probe.py`、
`tmp/mjx-ray-eval/dynamic_batch.py` 及相应 JSON。
官方源码安装路径：
`mujoco/mjx/_src/ray.py`（图元分派/材质索引）及
`mujoco/mjx/_src/io.py`（模型转换的碰撞校验）。
这段评估不对运行时引入第二份 `mjx.Model/Data`。

## 5. 从高保真真实性出发，重审目前已批准的快照采集

### 5.1 现有优势（已验证）

- **几何第一回波：**当前 Box、Cylinder、Sphere、Capsule、Plane
  的求交与 MuJoCo C 参考做过几何/随机旋转/动态时间验证；
  能复用相同 MJCF，区分感知可见性与碰撞净空。
- **设备扫描覆盖：**LiDAR 不是合成黄金比例方向，而是读取上游
  `mid360.npy`；固定 SHA-256，角度源、相位可追踪。
- **训练集成：**仍可进行 JAX/JIT 编译与 APG/SHAC 反传；
  六组 Depth/LiDAR × PPO/APG/SHAC 完成 GPU 更新与检查点恢复验证。
- **受控比较：**候选求交失败就不切换，避免改变几何正确性却把失败归因于策略算法。

### 5.2 高保真方面仍然欠缺的事实

| 项目 | 当前状态 | 高保真缺口 |
|---|---|---|
| MID360 角度 | 读第三方 800,000 对角度；约每 20,000 条组成 0.1 s 的逻辑帧 | 只有发布的角度资源，**没有找到实测标定/每点硬件时间戳出处**，不可宣称“已复刻真实光机运动” |
| MID360 扫描时间 | 一帧 20,000 条射线全部用同一采集位姿和时刻 | 高速/动态场景不会产生真实的扫描畸变 |
| 角度序列 | 当前每帧取 20,000 条，800,000 条循环 | 40 帧（4 秒）后**角度序列周期重复**；只是有限非重复扫描资源 |
| 距离、方向噪声 | `error_model=ideal` | 不模拟距离精度、角度扰动、失效回波、虚警、材料反射率 |
| 光路与形状 | 原始 MJCF 的基本解析图元 | 当前没有复杂 Mesh / HField；也不表示真实材料的激光反射 |
| D435i | 64×48 训练针孔深度、30 Hz，4×4 池化到 16×12 | 是**理想低分辨率深度模型**，没有主动立体测距的失效/光学与曝光噪声 |
| 遮挡 | 由场景可见几何计算 | 若机身/桨叶不在 `Scene` 可见几何中，就不会产生真实的机体自遮挡 |

Livox [MID-360 官方规格](https://www.livoxtech.com/cn/mid-360/specs)：
首回波 **200,000 points/s**、典型 **10 Hz**、水平 360°、
垂直 −7° 至 52°、近距盲区 0.1 m；
随机测距误差（1σ）在指定测试条件下 ≤2 cm @ 10 m、
角度随机误差小于 0.15°。这些是**硬件规格及条件上界**，
不是声称模拟器必须使用固定 2 cm 高斯噪声。

特别注意：无人机以 20 m/s 运动 0.1 s，可移动 **2 m**；
即便只有 3 m/s，也可移动 **0.3 m**。
因此，**“每 0.1 s 完成 20,000 个瞬时射线命中”是理想快照假设，
不是高保真 20 m/s MID360 扫描**。
早先将快照改造带来的约 7.7% Depth、11.1% LiDAR 完整训练吞吐收益
不能被解读为保持物理传感器真实性的无代价加速。

### 5.3 该如何调整先前决策？

**本次 Review 不建议回滚所有传感器改造**：扫描资源、配置显式化、
统一 `Measurement`、checkpoint 身份、低维护成本都是长期正确方向。
但应修正对外描述与后续验收目标：

1. **当前版本明确称为 Idealized MID360 / Ideal Depth**，
   不称为完整高保真传感器复刻；保持现有唯一求交实现。
2. **如果真正的研究目标是 20 m/s 实机感知避障，下一项独立研究优先恢复
   “扫描期间的采集时间语义”并做动态障碍/机体运动消融。**
   可以保持同一个 `Scene.raycast` 核心，但需给每射线/时间分片
   真实的场景位置与位姿；角度文件只有方向没有可核实的时间标定，
   时间分配必须清楚标注为均匀采样假设，而非厂商实测。
   该更改会改变观测分布、JAX 状态内存和训练吞吐，
   **需要新的 ADR 批准、独立冻结评测与从零训练验收**。
3. **真实物理传感器的误差建模要先于 Mesh 扩展决定。**
   可以按官方规格引入受控的距离/角度噪声、漏点和延迟对照，
   并保留完全相同的感知真值用于单元测试；但不得伪造
   未测量的反射率、雨雾和光机扫描误差。
4. **有明确复杂场景或视觉 Mesh 任务后**，
   再将官方 MuJoCo Warp / mjlab 的 BVH、Renderer 作为一个完整候选，
   验证 Cylinder、动态 Mesh、坐标/遮挡、GPU/JAX 交互、
   20,000 rays × 多世界 × 时间分片、
   500 Hz 物理与可微 APG/SHAC 更新；
   验收通过后**一次性替换**，删除自写求交内核，不保留多个永久 Backend。

## 6. 架构 Review：代码标准轴 × 研究目标轴

### A. Standards（维护性）

**已符合：**现有 `Scene.raycast(origin, directions, t, max_range)`
是窄的公共接口；几何与时间由 Scene 封装，
传感器只得到返回值，符合 *deep module with narrow interface* 原则。
现有真实实现只有一套，因此再增加 `backend=third_party`
注册器是没有必要的假设 seam。

**须避免：**将一台传感器拆成通用 RaycastProvider / PatternStrategy /
AdapterFactory 层，或让每种算法理解不同底层几何对象。
如果删除 `Scene.raycast` 会把“动态障碍位置、可见过滤、
最大距离、坐标转换”分散到 Depth、LiDAR、ROS、回放等调用方，
说明它目前是有效的深模块，而不是无意义封装。

### B. Spec（研究真实性）

**几何求交与算法梯度：基本达标。** 当前基本图元与 MuJoCo 的
第一回波合同有较广泛测试，采集梯度截断而动力学/策略梯度仍存在；
冻结策略的 Depth/LiDAR 多场景复测已有独立证据。

**高速 MID360 物理真实性：尚未达标。** 无帧内运动畸变、无可信硬件
时间标定、无真实噪声/反射/自遮挡，不能称为“实机级高保真”。
尤其不能用“用了第三方 LiDAR 模式表”替代设备级误差与时间合同测试。

**复杂 Mesh：当前不在目标范围。** 高保真不是功能堆叠：
当前的圆柱和盒体是解析几何真值；为了用 Mesh 引擎而把它们多边形化，
并不自动更真实。

## 7. 明确的替换门禁

将来只有候选**同时**达到以下结果，才应计划删除当前 JAX 求交实现：

- 以 MuJoCo C `mj_multiRay` 为几何参考，覆盖现有所有几何类型，
  包括 Cylinder、近场内部起点、切线、平面单/双面、透明和距离截断语义；
  不能以只核对数组维数替代。
- 同样的随机点位、动态场景时间、障碍物位姿、扫描角度下，对
  S01–S03、D01–D03、S06/D06 完成正确性/可见性核对；
  将几何误差和传感器时间/噪声差异分开统计。
- 在 RTX 4090 上比较相同批量和时间模型：
  LiDAR 32×32、每帧 20,000 条；Depth 128×32、64×48；
  报告纯求交、完整 PPO/APG/SHAC 更新、首次编译及显存，
  含 `jax.jit/lax.scan/checkpoint/grad`；不能借减少点数提高性能。
- 原冻结任务门槛和相同随机种子重测；如果感知分布改变，
  使用新的独立训练/评测协议，不将旧 checkpoint 测试集成绩
  用作反复选模依据。
- 验收后只保留一个实现，现有回放、Planner/ROS、训练网络无需理解求交器版本；
  GPU 队列同步与第三方模型内存须局部封装并可删除。

**最终判定：当前保留解析 JAX，不切换 MJX-JAX；`mujoco-lidar` 仅复用扫描资源。
高保真下一步优先确认扫描时间和实测误差，而不是换一个名字更成熟的求交库。**
