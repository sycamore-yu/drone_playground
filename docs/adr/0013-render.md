# ADR-0013: render

**Status:** Accepted

**批准记录时间:** 2026-10-10T08:28:44Z。用户批准本轮全部建议并要求实施。

**创建时间:** 2026-10-10T15:17:33+08:00

**最近核对时间:** 2026-10-10T10:30:23Z；代码和 GPU 运行证据见[验证记录](../validation.md)。

**代码基线:** `v0.2@6ec50c7a441bbe169fef40acd34ae574b53428a9`

**工作分支:** `refactor/sensor-rendering`

**实现状态:** 已将训练参数移入配置，深度与 LiDAR 共用现有单一路径；官方 Renderer 对照测试已加入。局部高精度几何修复后的 Depth/APG 在 S01、128×32 的 CUDA 完整更新吞吐实测提升约 7.7%，冻结策略在八场景取得 196/200。MuJoCo-LiDAR JAX 与官方 MJX-JAX 替换候选均未通过当前门禁，保留唯一 `Scene.raycast()`；未安装 Warp 或增加第二套后端。[研究与 Review](../research/sensor-raycasting-fidelity-review.md)。

## 背景

当前训练需要理想深度和 LiDAR，不需要 RGB、纹理或 Mesh 场景。已有深度图由 JAX GPU 射线求交生成，不能把它当作串行 CPU 渲染，再套用 GPU 渲染器的加速倍数。

本 ADR 中的 render 指从场景生成传感器图像。面向人的 RScope 回放仍使用现有回放路径，不和训练采样绑定。

**2026-10-11 RScope 展示改进：**由旧仓库复制 Crazyflie 2.x 原始 STL
和授权文件，只作为 Replay 的可见 Mesh；继续使用新模型中的
Freejoint 和已记录位姿。规划点除了命中点标记，还可按时间有效性显示
连续空间线段。点云仍来自运行时测量记录，不再求交；
详细 API、有效期和独立 wheel 测试见 [回放文档](../replay.md)。

## 决策

### 共用唯一求交内核，暂不加入 Warp

理想深度与 LiDAR 共用一套 `Scene.raycast()` 射线求交。
`MjLidarJax.render_batch()` 是已否决的直接替换候选，不在正式训练中运行；
官方 MJX-JAX `ray()` 也不覆盖当前大量 Cylinder 场景，
详见 [ADR-0012](0012-sensor.md) 和[求交 Review](../research/sensor-raycasting-fidelity-review.md)。
相机生成像素中心射线，将径向距离乘以射线在光轴方向的分量得到光轴深度，再应用量程和有效掩码。未命中判定先于深度转换，避免舍入将远裁剪值变成有效回波。

此处复用的是通用射线查询，并不将 LiDAR 扫描图伪装成深度图。相机像素中心、内外参、光轴方向和图像尺寸都独立确定。现有 `Measurement`、训练预处理、Actor 和梯度规则继续使用，不增加 Renderer 注册器或第二套环境。

官方 `mujoco.Renderer` 与 `mj_ray` 只作为测试参考，不作为可选训练后端。测试须对齐像素中心、深度定义、几何可见性及裁剪范围，不要求不同光栅化路径逐像素位相同。

本轮不安装或实现 MJX-Warp Batch Renderer。它支持批量 RGB/Depth 和复杂场景，但尚无本仓库相同工作负载下的速度证据。未来实测有收益时，把它作为下一次替换候选，而不是提前加入产品后端菜单。

### 将训练分辨率与设备标定分开

`64×48` 是训练图像分辨率，不是 D435i 的原生硬件模式。设备构造器已删除 `training64x48` 模式。`configs/sensor/depth.yaml` 明确声明宽高、30 Hz、20° 安装角及 10 m 仿真截止；网络预处理仍在原 Observation 配置中。Python 调用同样显式传入这些参数。

| 来源 | 图像宽×高 | 训练采集周期 | 依据 |
|---|---|---|---|
| Zhang / DiffPhysDrone，`271936190b5c` | `64×48`，再 4×4 max-pool 得到 `16×12` | `normalvariate(1/15, 0.1/15)` 秒，每步渲染，名义约 15 Hz | `main_cuda.py` |
| DiffAero，`291ea14196aefbebcf7387dd71f7e096c83878b7` 默认配置 | `16×9` | `env.dt=0.0333` 秒，障碍规避环境每步更新传感器，约 30 Hz | `cfg/sensor/camera.yaml`、`cfg/config_train.yaml`、`env/obstacle_avoidance.py` |
| 当前 Drone Playground 基线 | `64×48`，再池化为 `16×12` | Sensor 30 Hz；导航策略 10 Hz | `simulation/sensors.py` 与现有实验配置 |

当前组合的低分辨率预处理参考 Zhang；30 Hz 是项目的设备采样配置，不是 Zhang 的训练周期。DiffAero 的默认配置也不是 `64×48`。上述来源都说明可以直接用低分辨率图像训练，不需要先生成硬件最大分辨率再缩小。

验收继续使用 `64×48、30 Hz` 的明确实验配置；不增加旧构造模式兼容层。需要按论文条件训练时，在实验配置中采用对应周期和预处理并单独记录结果。本次没有更换 CNN 结构、池化尺寸或训练算法。

### 不为当前任务扩展 Mesh

现有导航与竞速场景由基本几何体构成，传感器求交没有已确认的 Mesh 需求。机器人外观显示由回放系统处理。本轮保持当前几何集合和可微净空代价，不为假设中的复杂场景新增 Mesh 管理、BVH 同步或碰撞代理层。

## 性能判据

本仓库的 MJX-Warp 加速倍数目前未知，甚至不能排除小分辨率下新增同步和调度成本抵消收益。官方批量渲染基准只说明其在所测场景的吞吐，不能代替和当前纯 JAX 求交的比较。

若采集占一次更新的时间比例为 `f`，新采集实现快 `s` 倍，忽略新增同步时，整个更新的理想加速为 `1 / ((1-f) + f/s)`。例如假设 `f=0.3、s=5`，整个更新约快 1.32 倍；这只是计算示例，不是本项目测量或预测。

未来只有 profiling 证实采集是重要瓶颈，且官方方案在相同分辨率、频率、射线、时间语义和批量上改善完整更新吞吐，才安排 Warp 迁移。测试包括预热、GPU 同步、显存和图捕获开销，以及实际 `jit/scan/checkpoint/grad` 调用。官方文档中的固定 `nworld` 与 `vmap(scan)` 限制需要对照实际调用结构验证，不能据此直接断言当前结构不能使用。

## 验收与删除

先验证针孔深度及范围转换，再运行 PPO/APG/SHAC 的真实更新和既定任务验收。几何与时间语义相同的替换采用相同输入进行数值对照；扫描模式或采样周期发生变化的实验按观测分布变更处理。

仅当将来另一个官方成熟候选通过 [ADR-0012](0012-sensor.md) 的全部门禁后，才一次性删除当前自写射线求交，不保留多套永久后端。正式测试保留官方参考，测试结果保存在现有结果目录。自定义模型距离函数、网络预处理和人用回放系统继续承担各自已有职责。

## 依据

- [Zhang 官方训练代码，固定版本](https://github.com/HenryHuYu/DiffPhysDrone/blob/271936190b5c/main_cuda.py)。
- [DiffAero 相机默认配置](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/cfg/sensor/camera.yaml)、[训练周期](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/cfg/config_train.yaml)、[采集调用](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/env/obstacle_avoidance.py)。
- [MuJoCo-LiDAR JAX 核心](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/core_jax/mjlidar_jax.py)。
- [官方 MJX 批量渲染文档](https://mujoco.readthedocs.io/en/latest/mjx.html#batch-rendering)、[官方 MJWarp 渲染及基准说明](https://mujoco.readthedocs.io/en/latest/mjwarp/#batch-rendering)。
- [现有传感器](../../src/drone_playground/simulation/sensors.py)、[现有性能记录](../performance.md)、[既有梯度说明](../training.md)。
