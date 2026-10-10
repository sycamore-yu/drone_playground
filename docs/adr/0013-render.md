# ADR-0013: render

**Status:** Proposed

**批准时间:** 未批准具体实现；用户已要求减少后端和重复实现。

**创建及最近核对时间:** 2026-10-10T15:17:33+08:00

**代码基线:** `v0.2@6ec50c7a441bbe169fef40acd34ae574b53428a9`

**工作分支:** `refactor/sensor-rendering`

**实现状态:** 文档提案；未接入新渲染器，未运行 GPU 性能或训练验收。

## 背景

当前训练需要理想深度和 LiDAR，不需要 RGB、纹理或 Mesh 场景。已有深度图由 JAX GPU 射线求交生成，不能把它当作串行 CPU 渲染，再套用 GPU 渲染器的加速倍数。

本 ADR 中的 render 指从场景生成传感器图像。面向人的 RScope 回放仍使用现有回放路径，不和训练采样绑定。

## 提议决策

### 共用原生求交，暂不加入 Warp

理想深度相机使用与 LiDAR 相同的 MuJoCo-LiDAR `MjLidarJax.render_batch()`。相机只需生成由内参和外参决定的针孔射线，将返回径向距离转换为光轴深度，再应用现有量程和有效掩码。对单位射线，光轴深度等于射线距离乘其在光轴方向的分量。

此处复用的是通用射线查询，并不将 LiDAR 扫描图伪装成深度图。相机像素中心、内外参、光轴方向和图像尺寸都独立确定。现有 `Measurement`、训练预处理、Actor 和梯度规则继续使用，不增加 Renderer 注册器或第二套环境。

官方 `mujoco.Renderer` 与 `mj_ray` 只作为测试参考，不作为可选训练后端。测试须对齐像素中心、深度定义、几何可见性及裁剪范围，不要求不同光栅化路径逐像素位相同。

本轮不安装或实现 MJX-Warp Batch Renderer。它支持批量 RGB/Depth 和复杂场景，但尚无本仓库相同工作负载下的速度证据。未来实测有收益时，把它作为下一次替换候选，而不是提前加入产品后端菜单。

### 将训练分辨率与设备标定分开

`64×48` 是训练图像分辨率，不是 D435i 的原生硬件模式。D435i 预设负责设备标定；图像尺寸、采集频率和网络预处理由已有 Sensor/Experiment 配置明确组合，不在设备构造器中隐藏 `training64x48` 的研究假设。

| 来源 | 图像宽×高 | 训练采集周期 | 依据 |
|---|---|---|---|
| Zhang / DiffPhysDrone，`271936190b5c` | `64×48`，再 4×4 max-pool 得到 `16×12` | `normalvariate(1/15, 0.1/15)` 秒，每步渲染，名义约 15 Hz | `main_cuda.py` |
| DiffAero，`291ea14196aefbebcf7387dd71f7e096c83878b7` 默认配置 | `16×9` | `env.dt=0.0333` 秒，障碍规避环境每步更新传感器，约 30 Hz | `cfg/sensor/camera.yaml`、`cfg/config_train.yaml`、`env/obstacle_avoidance.py` |
| 当前 Drone Playground 基线 | `64×48`，再池化为 `16×12` | Sensor 30 Hz；导航策略 10 Hz | `simulation/sensors.py` 与现有实验配置 |

当前组合的低分辨率预处理参考 Zhang；30 Hz 是项目的设备采样配置，不是 Zhang 的训练周期。DiffAero 的默认配置也不是 `64×48`。上述来源都说明可以直接用低分辨率图像训练，不需要先生成硬件最大分辨率再缩小。

纯求交替换验收暂用现有 `64×48、30 Hz` 作为固定工作负载，避免同时改变 CNN 输入与时序。该组合保留为一个明确的实验参数组合，不建设旧模式兼容层。需要按某篇论文训练时，在实验配置中采用相应周期和预处理，并单独记录结果；本次不静默切到 15 Hz，也不为未经验证的 16×9 输入更改网络。

### 不为当前任务扩展 Mesh

现有导航与竞速场景由基本几何体构成，传感器求交没有已确认的 Mesh 需求。机器人外观显示由回放系统处理。本轮保持当前几何集合和可微净空代价，不为假设中的复杂场景新增 Mesh 管理、BVH 同步或碰撞代理层。

## 性能判据

本仓库的 MJX-Warp 加速倍数目前未知，甚至不能排除小分辨率下新增同步和调度成本抵消收益。官方批量渲染基准只说明其在所测场景的吞吐，不能代替和当前纯 JAX 求交的比较。

若采集占一次更新的时间比例为 `f`，新采集实现快 `s` 倍，忽略新增同步时，整个更新的理想加速为 `1 / ((1-f) + f/s)`。例如假设 `f=0.3、s=5`，整个更新约快 1.32 倍；这只是计算示例，不是本项目测量或预测。

未来只有 profiling 证实采集是重要瓶颈，且官方方案在相同分辨率、频率、射线、时间语义和批量上改善完整更新吞吐，才安排 Warp 迁移。测试包括预热、GPU 同步、显存和图捕获开销，以及实际 `jit/scan/checkpoint/grad` 调用。官方文档中的固定 `nworld` 与 `vmap(scan)` 限制需要对照实际调用结构验证，不能据此直接断言当前结构不能使用。

## 验收与删除

先验证针孔深度及范围转换，再运行 PPO/APG/SHAC 的真实更新和既定任务验收。几何与时间语义相同的替换采用相同输入进行数值对照；扫描模式或采样周期发生变化的实验按观测分布变更处理。

替换通过后，与 [ADR-0012](0012-sensor.md) 一同删除自写射线求交及迁移后端分派。正式测试保留官方参考，测试结果保存在现有结果目录。自定义模型距离函数、网络预处理和人用回放系统继续承担各自已有职责。

## 依据

- [Zhang 官方训练代码，固定版本](https://github.com/HenryHuYu/DiffPhysDrone/blob/271936190b5c/main_cuda.py)。
- [DiffAero 相机默认配置](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/cfg/sensor/camera.yaml)、[训练周期](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/cfg/config_train.yaml)、[采集调用](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/env/obstacle_avoidance.py)。
- [MuJoCo-LiDAR JAX 核心](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/core_jax/mjlidar_jax.py)。
- [官方 MJX 批量渲染文档](https://mujoco.readthedocs.io/en/latest/mjx.html#batch-rendering)、[官方 MJWarp 渲染及基准说明](https://mujoco.readthedocs.io/en/latest/mjwarp/#batch-rendering)。
- [现有传感器](../../src/drone_playground/simulation/sensors.py)、[现有性能记录](../performance.md)、[既有梯度说明](../training.md)。
