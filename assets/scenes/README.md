# 场景资产

场景几何使用 **MJCF/XML**，由 Simulation 装配和执行。单机四旋翼的训练与评测按任务配置选择场景，运动障碍的位姿随仿真时间更新。

## 固定场景

| Task | 资产 | 研究用途 |
|---|---|---|
| Tracking | 空场景 | Hover、Figure-eight、随机样条等参考 |
| Racing | `racing/lsy_level0.xml`、`lsy_native.yaml` | LSY Level0，过门顺序 `1→2→3→4→2` |
| Navigation | `navigation/catalog.xml`、`boundary.xml`、8 张固定 MJCF | S01/S02/S03、D01/D02/D03 主场景；S06/D06 扩展场景 |

Navigation 的空间范围为 `[0,-20,0.5]` 至 `[100,20,6]` m，起点 `(2,0,3)` m，目标 `(98,0,3)` m。`catalog.xml` 保存场景来源与几何元数据，四面实体墙由 `boundary.xml` 定义。

## 动态障碍

障碍的 `body.user` 有六个数字：运动类型及五个参数。类型 `0=static`、`1=trefoil`、`2=linear_bounce`；`body.pos` 为静止位置或运动中心。场景运行时依据**仿真时间**更新 mocap 位姿。

Trefoil 参数 `(sx,sy,sz,offset,slower)`，设 `u=2*t/slower+offset`：

```text
dx = sx/6 * (sin(u) + 2*sin(2*u))
dy = sy/5 * (cos(u) - 2*cos(2*u))
dz = -sz/2 * sin(3*u)
```

Linear bounce 参数 `(ax,ay,az,period,phase)`，设 `u=t/period+phase`，`triangle=2*abs(2*(u-floor(u+0.5)))-1`，位姿为 `origin+(ax,ay,az)*triangle`。

D01/D02/D03 使用 trefoil；D06 包含往复运动障碍。碰撞、Depth/LiDAR 测量与 RScope 回放使用同一时刻的场景位姿。

## 素材来源

- **Navigation8**：基于 [SANDO](https://github.com/mit-acl/sando) 的森林与动态障碍构造，并按项目的固定边界、几何混合和种子建立 MJCF；场景具体参数见 `navigation/catalog.xml` 的 `catalog_metadata`。来源规则参考提交 `3a4450dcc5a8ed5ca825c9da7966e2642be091de`。这是具有独立几何身份的实验场景集合。
- **Racing**：来自 [LSY Drone Racing](https://github.com/learnsyslab/lsy_drone_racing) Level0，参考提交 `b1f5b36adb8e08e8e2adea85de790bd0e0a1d118`；纹理保存在 `racing/textures/`，许可证见 `racing/LICENSE`。
- **运行参考**：官方 [Crazyflow MuJoCo integration](https://learnsyslab.github.io/crazyflow/user-guide/mujoco/)；资产在打包后应可通过包资源加载。

实现验证：8 张 Navigation MJCF 和一张 Racing MJCF 可以用 MuJoCo 3.14.0 编译。动态运动、任务逻辑和 Crazyflow 集成按 [Simulation 验收](../../docs/spec.md) 的 S2–S4 检查。
