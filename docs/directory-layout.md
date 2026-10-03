# 目录职责

发布树把“可安装 Python 包”“非 Python 原生运行时”“第三方来源声明”和“研究产物”
分开。运行代码不依赖源码根目录的资源镜像。

## 根目录

| 目录 | 职责 |
|---|---|
| `src/drone_playground/` | 唯一可安装 Python 包；包含运行代码、Hydra 配置、benchmark 和运行资产 |
| `native/` | C++ SDK、ROS/Docker 等非 Python 构建与部署；不随 Python wheel 安装 |
| `third_party/` | 固定上游来源、revision 与仓库级补丁 |
| `scripts/` | 开发/维护工具；公开运行入口由 package console scripts 提供 |
| `tests/` | 单元、集成和回归测试 |
| `docs/` | 现役架构、操作、状态和研究说明；`docs/notes/archive/` 是历史快照 |
| `artifacts/` | 小型版本化验证证据与选择清单 |
| `results/` | 本地运行结果及选定结果引用 |
| `tmp/` | 可删除的依赖缓存、构建与一次性验证工作区 |

`build/` 与 `dist/` 是 setuptools/pip 生成物，已在 `.gitignore` 中忽略，不属于
仓库结构。

## `src/drone_playground/`

| 子目录/模块 | 职责 |
|---|---|
| `artifacts/` | checkpoint、run、trace、provenance、迁移与结果持久化 |
| `assets/` | wheel 内的 MJCF、mesh、texture 等物理/可视化资源 |
| `benchmarks/`, `benchmarks.py` | 版本化评测协议、固定几何身份与质量判定 |
| `configs/` | Hydra 配置组；YAML 是参数组合唯一来源 |
| `control/` | Reference 下游的 Setpoint/Actuation、controller、delay、decoder 与 transition |
| `dynamics/` | Crazyflow、LOTF、PointMass 等实际前向动力学和导数规则 |
| `environments/` | `DroneEnvironment`、scene、sensor、observation、task、reset/randomization |
| `evaluation/` | 冻结策略/方法的运行与任务指标 |
| `integrations/` | 随 Python 包安装的外部进程/ROS/gRPC 适配器 |
| `learning/` | PPO/BPTT/SHAC/DVA 等训练编排、wrapper 和冻结 inference |
| `networks/` | 网络 factory、感知编码、循环网络与冻结 policy wrapper |
| `planning/` | 进程内规划/参考模块和 SFC/preview 数据类型 |
| `rpc/` | 外部算法 transport、wire、protobuf 和生命周期；不实现 Planner |
| `runtime/` | Pipeline 调度、JAX/host runner、时钟、设备与运行时 decision |
| `visualization/` | RScope/MuJoCo replay、辅助显示层与 viewer |
| `references.py` | Waypoint / Trajectory 公共 Reference 表示 |
| `resources.py` | 通过 `importlib.resources` 定位 package data |
| `app.py`, `cli.py`, `composition.py`, `configuration.py` | Hydra/CLI 公共入口、配置解析和实验调度 |

源码包保持名称 `environments/`；`env` 是 Hydra 配置组名，两者不需要同名。

## 外部算法边界

`rpc/` 负责协议，`integrations/` 负责 Python 适配，
`native/` 负责 C++/ROS/Docker 工程。三者不是重复实现：

```
runtime Pipeline
    |
    +-- in-process Policy / Planner / Controller
    |
    +-- integrations/*
            |
            +-- rpc/* ---- native/sdk C++ process
            |
            +-- ros1.py -- native/ros1 ROS container
```

是否需要 native bridge 由部署边界决定，而不是由算法名称决定。仓库内 JAX/Flax
Policy（例如未来的 AllocateNet）直接返回声明的 Setpoint/Actuation，不需要 RPC；
只有把该方法作为独立 C++/ROS 进程运行时才需要 adapter。
